// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file DiffusionSelfAttention.cpp
 *  @brief Fused conditional self-attention used by the diffusion transformer.
 */

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/TaskPartition.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Attention.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::B;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::D;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "DiffusionSelfAttention requires an AMX or SME BF16 matmul target"
#endif

namespace {

#if defined(ARCH_X86_FAMILY)
constexpr nint_t RowsPerTask = decltype(N)::value <= 256 ? 16 : 32;
#else
constexpr nint_t RowsPerTask = 32;
#endif
using ActiveRows = meta::Dynamic<1, 1, RowsPerTask>;
using TileRows = std::conditional_t<(decltype(N)::value % RowsPerTask) == 0,
                                    meta::Const<RowsPerTask>, ActiveRows>;
using AttentionConfig =
    ops::AttentionConfig<ops::MatmulConfig<MatmulAtom>, 32, 32,
                         ops::AttentionCausalMode::none,
                         ops::SDPAStrategy::automatic, vec::Accuracy::Estimate>;

template <typename Tag, typename Value>
auto round_bf16(Tag tag, Value value) {
  using BF16Tag = vec::Rebind<vecops::bfloat16_t, Tag>;
  return vec::convert(tag, BF16Tag{},
                      vec::convert(BF16Tag{}, tag, value));
}

template <typename Tag, typename Value>
auto bounded_sigmoid(Tag tag, Value value) {
  const auto one = vec::fill(tag, 1.0f);
  const auto bounded = vec::max(
      tag, vec::fill(tag, -20.0f),
      vec::min(tag, value, vec::fill(tag, 20.0f)));
  const auto z = vec::mul(tag, vec::neg(tag, vec::abs(tag, bounded)),
                          vec::fill(tag, 1.0f / 256.0f));
  auto polynomial = vec::add(
      tag, vec::fill(tag, 1.0f / 24.0f),
      vec::mul(tag, z, vec::fill(tag, 1.0f / 120.0f)));
  polynomial = vec::add(tag, vec::fill(tag, 1.0f / 6.0f),
                        vec::mul(tag, z, polynomial));
  polynomial = vec::add(tag, vec::fill(tag, 0.5f),
                        vec::mul(tag, z, polynomial));
  polynomial = vec::add(tag, one, vec::mul(tag, z, polynomial));
  auto exp_abs = vec::add(tag, one, vec::mul(tag, z, polynomial));
  for (int i = 0; i < 8; ++i)
    exp_abs = vec::mul(tag, exp_abs, exp_abs);
  const auto denominator = vec::add(tag, one, exp_abs);
  return vec::blend(tag, vec::div(tag, exp_abs, denominator),
                    vec::cmpge(tag, bounded, vec::zeros(tag)),
                    vec::div(tag, one, denominator));
}

template <TensorLike Scale, TensorLike Bias>
auto adaptive_norm_transform(Scale scale, Bias bias) {
  const auto *scale_data = scale.data();
  const auto *bias_data = bias.data();
  const nint_t scale_stride = scale.stride(0);
  const nint_t bias_stride = bias.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [scale_data, bias_data, scale_stride, bias_stride](
          auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto active = context.active_option(tag);
        const auto scale_value =
            vec::load_convert(tag, scale_data + row * scale_stride + column,
                              active, vec::opt::zero);
        const auto bias_value =
            vec::load_convert(tag, bias_data + row * bias_stride + column,
                              active, vec::opt::zero);
        const auto sigmoid = round_bf16(tag, bounded_sigmoid(tag, scale_value));
        const auto normalized = round_bf16(tag, value);
        const auto scaled = round_bf16(tag, vec::mul(tag, sigmoid, normalized));
        return vec::add(tag, scaled, bias_value);
      });
}

template <TensorLike Gate> auto sigmoid_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            const auto coordinate = context.lane_coord(0);
            const nint_t row = coordinate[0];
            const nint_t column = coordinate[1];
            const auto gate_value =
                vec::load_convert(tag, gate_data + row * gate_stride + column,
                                  context.active_option(tag), vec::opt::zero);
            const auto sigmoid = round_bf16(tag, bounded_sigmoid(tag, gate_value));
            return round_bf16(tag, vec::mul(tag, round_bf16(tag, value), sigmoid));
          });
}

template <TensorLike Gate, TensorLike Residual>
auto sigmoid_gate_residual_transform(Gate gate, Residual residual) {
  const auto *gate_data = gate.data();
  const auto *residual_data = residual.data();
  const nint_t gate_stride = gate.stride(0);
  const nint_t residual_stride = residual.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, residual_data, gate_stride, residual_stride](
          auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto gate_value =
            vec::load_convert(tag, gate_data + row * gate_stride + column,
                              context.active_option(tag), vec::opt::zero);
        const auto sigmoid = round_bf16(tag, bounded_sigmoid(tag, gate_value));
        const auto gated = round_bf16(
            tag, vec::mul(tag, round_bf16(tag, value), sigmoid));
        const auto residual_value = vec::load_convert(
            tag, residual_data + row * residual_stride + column,
            context.active_option(tag), vec::opt::zero);
        return vec::add(tag, gated, residual_value);
      });
}

template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, nint_t index, Spatial spatial,
              Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout),
                "packed B operand is too small");
  return make_tensor(packed.data() + index * packed.stride(0), layout);
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto key_mask, TensorLike auto pair_logits,
                TensorLike auto adaptive_scale, TensorLike auto adaptive_bias,
                TensorLike auto zero_gate, TensorLike auto q_projection,
                TensorLike auto q_bias, TensorLike auto k_projection,
                TensorLike auto v_projection, TensorLike auto gate_projection,
                TensorLike auto output_projection, TensorLike auto output) {
  assert_shape(act, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(key_mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(pair_logits, {vecops::any, H, N, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(adaptive_scale, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(adaptive_bias, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(zero_gate, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(q_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(q_bias, {H, D}, {D, cint<1>});
  assert_shape(k_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(v_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(gate_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  static_assert(decltype(H * D)::value == decltype(C)::value,
                "SelfAttention requires H*D == C");

  const auto batch = BatchExtent{act.size(0)};
  const nint_t batch_count = act.size(0);
  const nint_t mask_batch_count = key_mask.size(0);
  const nint_t pair_batch_count = pair_logits.size(0);
  const nint_t cond_batch_count = adaptive_scale.size(0);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  VECOPS_ASSERT(batch_count % mask_batch_count == 0 &&
                    batch_count % pair_batch_count == 0 &&
                    batch_count % cond_batch_count == 0,
                "shared batch extents must divide the activation batch");
  if (batch_count == 0 || sequence == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  constexpr auto TileCount = ceil_div(N, cint<RowsPerTask>);
  const nint_t tiles = static_cast<nint_t>(TileCount);
  const TileRows tile_rows{std::min(sequence, RowsPerTask)};

  const auto row_tasks = batch * TileCount;
  // Compare the fused row-tile decomposition with the head-parallel phases.
  // Work units describe the repeated inner operations; the planner accounts
  // for every partial worker wave instead of relying on a task-count threshold.
  constexpr auto HeadCount = H;
  const auto coarse = execution::task_partition_candidate(
      cint<0>,
      execution::task_partition_phase(row_tasks, cint<4> * HeadCount + cint<2>),
      execution::task_partition_phase(row_tasks, HeadCount + cint<1>));
  const auto fine = execution::task_partition_candidate(
      cint<2>, execution::task_partition_phase(row_tasks, cint<1>),
      execution::task_partition_phase(row_tasks * HeadCount, cint<5>),
      execution::task_partition_phase(row_tasks * HeadCount, cint<1>),
      execution::task_partition_phase(row_tasks, cint<1>));
  const auto candidates = std::tuple{coarse, fine};
  const auto partition =
      execution::choose_task_partition(cint<Parallelism>, candidates);
  if (partition.candidate_index == cint<0>) {
    const auto head_shape = make_shape(batch, H, N, D);
    auto global_scope = workspace.serial_scope("coarse_global");
    auto q = workspace.tensor<vecops::bfloat16_t>("q", head_shape);
    auto k = workspace.tensor<vecops::bfloat16_t>("k", head_shape);
    auto v = workspace.tensor<vecops::bfloat16_t>("v", head_shape);
    auto gate = workspace.tensor<vecops::bfloat16_t>("gate", head_shape);

    auto layer_norm = ops::layer_norm(
        ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
    auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
    auto pack_a_op = ops::matmul_pack(
        ops::MatmulPackConfig<MatmulAtom, matmul::Operand::A>{});
    auto attention_op = ops::scaled_dot_product_attention(AttentionConfig{});
    auto output_projection_t = packed_b(output_projection, 0, C, C);
    const auto channel_shape = make_shape(tile_rows, C);
    const nint_t tasks = static_cast<nint_t>(row_tasks);
    const auto packed_adapted_layout =
        matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
            make_layout(channel_shape));
    auto projection_scope = workspace.serial_scope("coarse_projection");
    auto adapted_values =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("adapted",
                                                                 channel_shape);
    auto packed_adapted_values =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "packed_adapted", packed_adapted_layout);
    auto sample_adapted = adapted_values(0, reserve, reserve);
    auto sample_packed_adapted = packed_adapted_values(0, ellipsis);
    auto sample_q_bias =
        make_tensor(q_bias.data(), make_layout(make_shape(tile_rows, D),
                                               make_strides(cint<0>, cint<1>)));
    auto prepared_norm = layer_norm.template prepare<Parallelism>(
        workspace, "norm_scratch",
        tensor::unbind(
            tensor::input<ComputeType>(act(0, range(0, tile_rows), reserve))),
        tensor::unbind(tensor::output<ComputeType>(
            sample_adapted,
            adaptive_norm_transform(
                adaptive_scale(0, range(0, tile_rows), reserve),
                adaptive_bias(0, range(0, tile_rows), reserve)))));
    auto prepared_pack = pack_a_op.template prepare<Parallelism>(
        workspace, "pack_scratch", tensor::unbind(sample_adapted),
        tensor::unbind(sample_packed_adapted));
    auto prepared_q_projection = matmul_op.template prepare<Parallelism>(
        workspace, "q_scratch", tile_rows, D, C,
        tensor::unbind(sample_packed_adapted),
        tensor::unbind(packed_b(q_projection, 0, D, C)),
        tensor::unbind(sample_q_bias),
        tensor::unbind(q(0, 0, range(0, tile_rows), reserve)));
    auto prepared_kvg_projection = matmul_op.template prepare<Parallelism>(
        workspace, "kvg_scratch", tile_rows, D, C,
        tensor::unbind(sample_packed_adapted),
        tensor::unbind(packed_b(k_projection, 0, D, C)),
        tensor::unbind(k(0, 0, range(0, tile_rows), reserve)));
    workspace.parallel_for<Parallelism>(
        nint_t{0}, tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          auto adapted = task_context.local(adapted_values);
          auto packed_adapted_storage =
              task_context.local(packed_adapted_values);
          const nint_t task = work_item.begin();
          const nint_t batch_index = task / tiles;
          const nint_t tile = task % tiles;
          const nint_t begin = tile * RowsPerTask;
          const nint_t end = std::min(begin + RowsPerTask, sequence);
          const TileRows rows{end - begin};
          const nint_t cond_index = batch_index % cond_batch_count;
          auto active_adapted = adapted(range(0, rows), reserve);
          prepared_norm(
              task_context, act(batch_index, range(begin, end), reserve),
              tensor::output<ComputeType>(
                  active_adapted,
                  adaptive_norm_transform(
                      adaptive_scale(cond_index, range(begin, end), reserve),
                      adaptive_bias(cond_index, range(begin, end), reserve))));
          auto active_packed_adapted_layout =
              matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
                  active_adapted.layout());
          auto active_packed_adapted = make_tensor(
              packed_adapted_storage.data(), active_packed_adapted_layout);
          prepared_pack(task_context, active_adapted, active_packed_adapted);
          for (nint_t head = 0; head < heads; ++head) {
            auto active_q_bias =
                make_tensor(q_bias.data() + head * q_bias.stride(0),
                            make_layout(make_shape(rows, D),
                                        make_strides(cint<0>, cint<1>)));
            prepared_q_projection(
                task_context, rows, D, C, active_packed_adapted,
                packed_b(q_projection, head, D, C), active_q_bias,
                q(batch_index, head, range(begin, end), reserve));
            prepared_kvg_projection(
                task_context, rows, D, C, active_packed_adapted,
                packed_b(k_projection, head, D, C),
                k(batch_index, head, range(begin, end), reserve));
            prepared_kvg_projection(
                task_context, rows, D, C, active_packed_adapted,
                packed_b(v_projection, head, D, C),
                v(batch_index, head, range(begin, end), reserve));
            prepared_kvg_projection(
                task_context, rows, D, C, active_packed_adapted,
                packed_b(gate_projection, head, D, C),
                gate(batch_index, head, range(begin, end), reserve));
          }
        });
    projection_scope.close();

    auto update_scope = workspace.serial_scope("coarse_update");
    const auto weighted_shape = make_shape(tile_rows, H, D);
    auto weighted_values =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "weighted", weighted_shape);
    auto sample_weighted = weighted_values(0, reserve, reserve, reserve);
    auto sample_weighted_flat =
        make_tensor(sample_weighted.data(), make_shape(tile_rows, C));
    const ComputeType attention_scale = static_cast<ComputeType>(
        1.0 / std::sqrt(static_cast<double>(head_dim)));
    auto prepared_attention = attention_op.template prepare<Parallelism>(
        workspace, "attention_scratch",
        tensor::unbind(q(0, 0, range(0, tile_rows), reserve)),
        tensor::unbind(k(0, 0, reserve, reserve)),
        tensor::unbind(v(0, 0, reserve, reserve)), tensor::nullopt,
        tensor::unbind(key_mask(0, reserve)), tensor::nullopt,
        tensor::unbind(pair_logits(0, 0, range(0, tile_rows), reserve)),
        tensor::unbind(tensor::output<ComputeType>(
            sample_weighted(reserve, 0, reserve),
            sigmoid_gate_transform(gate(0, 0, range(0, tile_rows), reserve)))));
    auto prepared_output = matmul_op.template prepare<Parallelism>(
        workspace, "output_scratch", tile_rows, C, C,
        tensor::unbind(sample_weighted_flat),
        tensor::unbind(output_projection_t),
        tensor::unbind(tensor::output<ComputeType>(
            output(0, range(0, tile_rows), reserve),
            sigmoid_gate_residual_transform(
                zero_gate(0, range(0, tile_rows), reserve),
                act(0, range(0, tile_rows), reserve)))));
    workspace.parallel_lanes<Parallelism>(
        [&](execution::TaskContext<Parallelism> task_context) {
          auto weighted = task_context.local(weighted_values);
          for (nint_t task = task_context.lane_id(); task < tasks;
               task += task_context.lane_count()) {
            const nint_t batch_index = task / tiles;
            const nint_t tile = task % tiles;
            const nint_t begin = tile * RowsPerTask;
            const nint_t end = std::min(begin + RowsPerTask, sequence);
            const TileRows rows{end - begin};
            auto active_weighted = weighted(range(0, rows), reserve, reserve);
            for (nint_t head = 0; head < heads; ++head) {
              prepared_attention(
                  task_context,
                  q(batch_index, head, range(begin, end), reserve),
                  k(batch_index, head, reserve, reserve),
                  v(batch_index, head, reserve, reserve), tensor::nullopt,
                  key_mask(batch_index % mask_batch_count, reserve),
                  tensor::nullopt,
                  pair_logits(batch_index % pair_batch_count, head,
                              range(begin, end), reserve),
                  tensor::output<ComputeType>(
                      active_weighted(reserve, head, reserve),
                      sigmoid_gate_transform(
                          gate(batch_index, head, range(begin, end), reserve))),
                  attention_scale);
            }
            auto active_weighted_flat =
                make_tensor(active_weighted.data(), make_shape(rows, C));
            const nint_t cond_index = batch_index % cond_batch_count;
            prepared_output(
                task_context, rows, C, C, active_weighted_flat,
                output_projection_t,
                tensor::output<ComputeType>(
                    output(batch_index, range(begin, end), reserve),
                    sigmoid_gate_residual_transform(
                        zero_gate(cond_index, range(begin, end), reserve),
                        act(batch_index, range(begin, end), reserve))));
          }
        });
    return;
  }

  // Keep normalized activations and head outputs global so the expensive
  // projection and attention phases can use independent (row tile, head)
  // tasks.  For one sample and N=512 this changes the main phases from 16
  // tasks to 256 tasks, enough to fill 32--64 core machines.
  const auto adapted_shape = make_shape(batch, N, C);
  const auto head_shape = make_shape(batch, H, N, D);
  const auto weighted_shape = make_shape(batch, N, H, D);
  auto global_scope = workspace.serial_scope("global");
  auto adapted = workspace.tensor<vecops::bfloat16_t>("adapted", adapted_shape);
  auto q = workspace.tensor<vecops::bfloat16_t>("q", head_shape);
  auto k = workspace.tensor<vecops::bfloat16_t>("k", head_shape);
  auto v = workspace.tensor<vecops::bfloat16_t>("v", head_shape);
  auto gate = workspace.tensor<vecops::bfloat16_t>("gate", head_shape);
  auto weighted =
      workspace.tensor<vecops::bfloat16_t>("weighted", weighted_shape);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto attention_op = ops::scaled_dot_product_attention(AttentionConfig{});
  auto output_projection_t = packed_b(output_projection, 0, C, C);

  // Domain 1: activation normalization, [B, tile(N)].
  const nint_t norm_tasks = batch_count * tiles;
  auto norm_scope = workspace.serial_scope("normalization");
  auto sample_act = act(0, range(0, tile_rows), reserve);
  auto sample_scale = adaptive_scale(0, range(0, tile_rows), reserve);
  auto sample_bias_state = adaptive_bias(0, range(0, tile_rows), reserve);
  auto sample_adapted = adapted(0, range(0, tile_rows), reserve);
  auto prepared_norm = layer_norm.template prepare<Parallelism>(
      workspace, "scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_act)),
      tensor::unbind(tensor::output<ComputeType>(
          sample_adapted,
          adaptive_norm_transform(sample_scale, sample_bias_state))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, norm_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch_index = task / tiles;
        const nint_t tile = task % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const nint_t cond_index = batch_index % cond_batch_count;
        prepared_norm(
            task_context, act(batch_index, range(begin, end), reserve),
            tensor::output<ComputeType>(
                adapted(batch_index, range(begin, end), reserve),
                adaptive_norm_transform(
                    adaptive_scale(cond_index, range(begin, end), reserve),
                    adaptive_bias(cond_index, range(begin, end), reserve))));
      });
  norm_scope.close();

  // Domain 2: Q/K/V/gate projections, [B, tile(N), H].
  const nint_t projection_tasks = batch_count * tiles * heads;
  auto projection_scope = workspace.serial_scope("head_projection");
  auto sample_projection = q(0, 0, range(0, tile_rows), reserve);
  auto q_bias_matrix =
      make_tensor(q_bias.data(), make_layout(make_shape(tile_rows, D),
                                             make_strides(cint<0>, cint<1>)));
  auto prepared_q_projection = matmul_op.template prepare<Parallelism>(
      workspace, "q_scratch", tile_rows, D, C,
      tensor::unbind(adapted(0, range(0, tile_rows), reserve)),
      tensor::unbind(packed_b(q_projection, 0, D, C)),
      tensor::unbind(q_bias_matrix), tensor::unbind(sample_projection));
  auto prepared_kvg_projection = matmul_op.template prepare<Parallelism>(
      workspace, "kvg_scratch", tile_rows, D, C,
      tensor::unbind(adapted(0, range(0, tile_rows), reserve)),
      tensor::unbind(packed_b(k_projection, 0, D, C)),
      tensor::unbind(sample_projection));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, projection_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t head = task % heads;
        const nint_t batch_tile = task / heads;
        const nint_t batch_index = batch_tile / tiles;
        const nint_t tile = batch_tile % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const TileRows rows{end - begin};
        auto active_adapted = adapted(batch_index, range(begin, end), reserve);
        auto active_q_bias = make_tensor(
            q_bias.data() + head * q_bias.stride(0),
            make_layout(make_shape(rows, D), make_strides(cint<0>, cint<1>)));
        prepared_q_projection(task_context, rows, D, C, active_adapted,
                              packed_b(q_projection, head, D, C), active_q_bias,
                              q(batch_index, head, range(begin, end), reserve));
        prepared_kvg_projection(
            task_context, rows, D, C, active_adapted,
            packed_b(k_projection, head, D, C),
            k(batch_index, head, range(begin, end), reserve));
        prepared_kvg_projection(
            task_context, rows, D, C, active_adapted,
            packed_b(v_projection, head, D, C),
            v(batch_index, head, range(begin, end), reserve));
        prepared_kvg_projection(
            task_context, rows, D, C, active_adapted,
            packed_b(gate_projection, head, D, C),
            gate(batch_index, head, range(begin, end), reserve));
      });
  projection_scope.close();

  // Domain 3: attention, [B, tile(N), H].  Each task owns one head slice.
  const nint_t attention_tasks = batch_count * tiles * heads;
  auto attention_scope = workspace.serial_scope("attention_heads");
  const ComputeType attention_scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto sample_q = q(0, 0, range(0, tile_rows), reserve);
  auto sample_k = k(0, 0, reserve, reserve);
  auto sample_v = v(0, 0, reserve, reserve);
  auto sample_gate = gate(0, 0, range(0, tile_rows), reserve);
  auto sample_mask = key_mask(0, reserve);
  auto sample_pair = pair_logits(0, 0, range(0, tile_rows), reserve);
  auto sample_weighted_head = weighted(0, range(0, tile_rows), 0, reserve);
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "scratch", tensor::unbind(sample_q), tensor::unbind(sample_k),
      tensor::unbind(sample_v), tensor::nullopt, tensor::unbind(sample_mask),
      tensor::nullopt, tensor::unbind(sample_pair),
      tensor::unbind(tensor::output<ComputeType>(
          sample_weighted_head, sigmoid_gate_transform(sample_gate))));
  workspace.parallel_lanes<Parallelism>(
      [&](execution::TaskContext<Parallelism> task_context) {
        for (nint_t task = task_context.lane_id(); task < attention_tasks;
             task += task_context.lane_count()) {
          const nint_t head = task % heads;
          const nint_t batch_tile = task / heads;
          const nint_t batch_index = batch_tile / tiles;
          const nint_t tile = batch_tile % tiles;
          const nint_t begin = tile * RowsPerTask;
          const nint_t end = std::min(begin + RowsPerTask, sequence);
          prepared_attention(
              task_context, q(batch_index, head, range(begin, end), reserve),
              k(batch_index, head, reserve, reserve),
              v(batch_index, head, reserve, reserve), tensor::nullopt,
              key_mask(batch_index % mask_batch_count, reserve),
              tensor::nullopt,
              pair_logits(batch_index % pair_batch_count, head,
                          range(begin, end), reserve),
              tensor::output<ComputeType>(
                  weighted(batch_index, range(begin, end), head, reserve),
                  sigmoid_gate_transform(
                      gate(batch_index, head, range(begin, end), reserve))),
              attention_scale);
        }
      });
  attention_scope.close();

  // Domain 4: output projection and residual, [B, tile(N)].
  const nint_t output_tasks = batch_count * tiles;
  auto output_scope = workspace.serial_scope("output_projection");
  auto sample_weighted = weighted(0, range(0, tile_rows), reserve, reserve);
  auto sample_weighted_flat =
      make_tensor(sample_weighted.data(), make_shape(tile_rows, C));
  auto sample_zero_gate = zero_gate(0, range(0, tile_rows), reserve);
  auto sample_residual = act(0, range(0, tile_rows), reserve);
  auto sample_output = output(0, range(0, tile_rows), reserve);
  auto prepared_output = matmul_op.template prepare<Parallelism>(
      workspace, "scratch", tile_rows, C, C,
      tensor::unbind(sample_weighted_flat), tensor::unbind(output_projection_t),
      tensor::unbind(tensor::output<ComputeType>(
          sample_output,
          sigmoid_gate_residual_transform(sample_zero_gate, sample_residual))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, output_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch_index = task / tiles;
        const nint_t tile = task % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const TileRows rows{end - begin};
        auto active_weighted =
            weighted(batch_index, range(begin, end), reserve, reserve);
        auto active_weighted_flat =
            make_tensor(active_weighted.data(), make_shape(rows, C));
        const nint_t cond_index = batch_index % cond_batch_count;
        prepared_output(
            task_context, rows, C, C, active_weighted_flat, output_projection_t,
            tensor::output<ComputeType>(
                output(batch_index, range(begin, end), reserve),
                sigmoid_gate_residual_transform(
                    zero_gate(cond_index, range(begin, end), reserve),
                    act(batch_index, range(begin, end), reserve))));
      });
}
