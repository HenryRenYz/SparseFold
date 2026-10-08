// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file EvoformerSingleActUpdate.cpp @brief Fused AlphaFold 3 single update.
 */

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Attention.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
constexpr auto N = spec::N;
constexpr auto Q = spec::Q;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::D;
constexpr auto M = spec::Hidden;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "EvoformerSingleActUpdate requires an AMX or SME BF16 matmul target"
#endif

namespace {

constexpr nint_t RowsPerTask = 32;
constexpr nint_t MaxTileRows =
    decltype(N)::value < RowsPerTask ? decltype(N)::value : RowsPerTask;
using ActiveRows = meta::Dynamic<1, 1, MaxTileRows>;
using TileRows = std::conditional_t<(decltype(N)::value <= RowsPerTask) ||
                                        (decltype(N)::value % RowsPerTask == 0),
                                    meta::Const<MaxTileRows>, ActiveRows>;
using AttentionConfig =
    ops::AttentionConfig<ops::MatmulConfig<MatmulAtom>, 32, 32,
                         ops::AttentionCausalMode::none,
                         ops::SDPAStrategy::automatic, vec::Accuracy::Estimate>;

template <TensorLike Gate> auto sigmoid_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            VECOPS_ASSERT(context.vector_axis == 1,
                          "attention gate epilogue must traverse channels");
            const auto coordinate = context.lane_coord(0);
            const nint_t row = coordinate[0];
            const nint_t column = coordinate[1];
            const auto gate_value =
                vec::load_convert(tag, gate_data + row * gate_stride + column,
                                  context.active_option(tag), vec::opt::zero);
            const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
                tag, gate_value);
            return vec::mul(tag, value, sigmoid);
          });
}

auto silu_transform() {
  return make_elementwise_vec_transform<ComputeType, ComputeType>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(tag, value);
        return vec::mul(tag, value, sigmoid);
      });
}

template <TensorLike Gate> auto transition_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            VECOPS_ASSERT(context.vector_axis == 1,
                          "transition gate epilogue must traverse columns");
            const auto coordinate = context.lane_coord(0);
            const nint_t row = coordinate[0];
            const nint_t column = coordinate[1];
            const auto old_gate =
                vec::load_convert(tag, gate_data + row * gate_stride + column,
                                  context.active_option(tag), vec::opt::zero);
            return vec::mul(tag, old_gate, value);
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

void __kernel__(
    execution::WorkspaceContext &workspace, TensorLike auto pair_act,
    TensorLike auto single_act, TensorLike auto key_padding_mask,
    TensorLike auto pair_norm_scale, TensorLike auto pair_norm_offset,
    TensorLike auto logits_projection, TensorLike auto attention_norm_scale,
    TensorLike auto attention_norm_offset, TensorLike auto q_projection,
    TensorLike auto q_bias, TensorLike auto k_projection,
    TensorLike auto v_projection, TensorLike auto g_projection,
    TensorLike auto o_projection, TensorLike auto transition_norm_scale,
    TensorLike auto transition_norm_offset, TensorLike auto transition1,
    TensorLike auto transition2, TensorLike auto output) {
  assert_shape(pair_act, {vecops::any, N, N, Q},
               {vecops::any, vecops::any, Q, cint<1>});
  assert_shape(single_act, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(key_padding_mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(pair_norm_scale, {Q}, {cint<1>});
  assert_shape(pair_norm_offset, {Q}, {cint<1>});
  assert_shape(logits_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(attention_norm_scale, {C}, {cint<1>});
  assert_shape(attention_norm_offset, {C}, {cint<1>});
  assert_shape(q_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(q_bias, {H, D}, {D, cint<1>});
  assert_shape(k_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(v_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(g_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(o_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(transition_norm_scale, {C}, {cint<1>});
  assert_shape(transition_norm_offset, {C}, {cint<1>});
  assert_shape(transition1, {cint<2>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(transition2, {cint<1>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, N, C}, {vecops::any, C, cint<1>});

  const auto batch_extent = BatchExtent{pair_act.size(0)};
  const nint_t batch_count = pair_act.size(0);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t pair_channels = static_cast<nint_t>(Q);
  const nint_t channels = static_cast<nint_t>(C);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  const nint_t hidden = static_cast<nint_t>(M);
  if (batch_count == 0 || sequence == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const auto pair_logits_shape = make_shape(batch_extent, H, N, N);
  const auto normalized_shape = make_shape(batch_extent, N, C);
  const auto head_shape = make_shape(batch_extent, H, N, D);
  auto global_lifetime = workspace.serial_scope("global");
  auto pair_logits = workspace.tensor<vecops::bfloat16_t>(
      "pair_logits", pair_logits_shape);
  auto normalized = workspace.tensor<vecops::bfloat16_t>(
      "normalized", normalized_shape);
  auto q = workspace.tensor<vecops::bfloat16_t>("q", head_shape);
  auto k = workspace.tensor<vecops::bfloat16_t>("k", head_shape);
  auto v = workspace.tensor<vecops::bfloat16_t>("v", head_shape);
  auto gate = workspace.tensor<vecops::bfloat16_t>("gate", head_shape);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto attention_op = ops::scaled_dot_product_attention(AttentionConfig{});
  auto logits_projection_b = packed_b(logits_projection, 0, H, Q);
  auto transition1_gate = packed_b(transition1, 0, M, C);
  auto transition1_value = packed_b(transition1, 1, M, C);
  auto transition2_b = packed_b(transition2, 0, C, M);

  // Pair bias projection. Each task owns one complete [N,Q] pair row, so the
  // row-wise layer norm is performed by the LayerNorm operator itself.
  const nint_t pair_tasks = batch_count * sequence;
  const auto norm_shape = make_shape(N, Q);
  auto pair_phase = workspace.serial_scope("pair_bias");
  auto pair_norm_workers =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "pair_norm", norm_shape);
  auto pair_norm_pattern = pair_norm_workers(0, ellipsis);
  auto sample_pair = pair_act(0, 0, reserve, reserve);
  auto sample_logits = transpose<0, 1>(pair_logits(0, reserve, 0, reserve));
  auto prepared_pair_norm = layer_norm.template prepare<Parallelism>(
      workspace, "pair_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_pair)),
      tensor::unbind(tensor::input<ComputeType>(pair_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(pair_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(pair_norm_pattern)));
  auto prepared_logits = matmul_op.template prepare<Parallelism>(
      workspace, "pair_logits_scratch", N, H, Q,
      tensor::unbind(pair_norm_pattern), tensor::unbind(logits_projection_b),
      tensor::unbind(sample_logits));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, pair_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / sequence;
        const nint_t row = task % sequence;
        auto active_pair = pair_act(batch, row, reserve, reserve);
        auto active_logits =
            transpose<0, 1>(pair_logits(batch, reserve, row, reserve));
        auto pair_norm = task_context.local(pair_norm_workers);
        prepared_pair_norm(task_context, active_pair, pair_norm_scale,
                           pair_norm_offset, pair_norm);
        prepared_logits(task_context, pair_norm, logits_projection_b,
                        active_logits);
      });
  pair_phase.close();

  // Normalize single activations by row tile first. Projection work is split
  // independently over heads below, so N=32 exposes H tasks instead of one.
  const nint_t tiles = ceil_div(sequence, RowsPerTask);
  if constexpr (decltype(N)::value < 64) {
    const nint_t normalization_tasks = batch_count * tiles;
    const TileRows tile_rows{std::min(sequence, RowsPerTask)};
    auto sample_single = single_act(0, range(0, tile_rows), reserve);
    auto sample_norm = normalized(0, range(0, tile_rows), reserve);
    auto normalization_phase = workspace.serial_scope("single_norm");
    auto prepared_attention_norm = layer_norm.template prepare<Parallelism>(
        workspace, "attention_norm_scratch",
        tensor::unbind(tensor::input<ComputeType>(sample_single)),
        tensor::unbind(tensor::input<ComputeType>(attention_norm_scale)),
        tensor::unbind(tensor::input<ComputeType>(attention_norm_offset)),
        tensor::unbind(tensor::output<ComputeType>(sample_norm)));
    workspace.parallel_for<Parallelism>(
        nint_t{0}, normalization_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          const nint_t batch = task / tiles;
          const nint_t tile = task % tiles;
          const nint_t begin = tile * RowsPerTask;
          const nint_t end = std::min(begin + RowsPerTask, sequence);
          prepared_attention_norm(
              task_context, single_act(batch, range(begin, end), reserve),
              attention_norm_scale, attention_norm_offset,
              normalized(batch, range(begin, end), reserve));
        });
    normalization_phase.close();

    // Create Q/K/V/gate with one (batch,tile,head) task. Q bias is represented
    // as a zero-row-stride matrix and therefore enters the accumulator.
    const nint_t projection_tasks = normalization_tasks * heads;
    const nint_t available_projection_threads = workspace.parallelism();
    const nint_t projection_threads =
        available_projection_threads <= 8
            ? std::min(projection_tasks, available_projection_threads)
            : nint_t{1};
    auto sample_q = q(0, 0, range(0, tile_rows), reserve);
    auto sample_q_bias =
        make_tensor(q_bias.data(), make_layout(make_shape(tile_rows, D),
                                               make_strides(cint<0>, cint<1>)));
    auto sample_k = k(0, 0, range(0, tile_rows), reserve);
    auto projection_phase = workspace.serial_scope("head_projection");
    auto prepared_q = matmul_op.template prepare<Parallelism>(
        workspace, "q_projection_scratch", tile_rows, D, C,
        tensor::unbind(sample_norm),
        tensor::unbind(packed_b(q_projection, 0, D, C)),
        tensor::unbind(sample_q_bias), tensor::unbind(sample_q));
    auto prepared_projection = matmul_op.template prepare<Parallelism>(
        workspace, "projection_scratch", tile_rows, D, C,
        tensor::unbind(sample_norm),
        tensor::unbind(packed_b(k_projection, 0, D, C)),
        tensor::unbind(sample_k));
    const nint_t projection_chunk =
        projection_threads == 1 ? projection_tasks : nint_t{1};
    workspace.parallel_for<Parallelism>(
        nint_t{0}, projection_tasks, projection_chunk,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          for (nint_t task = work_item.begin(); task < work_item.end(); ++task) {
            const nint_t matrix = task / heads;
            const nint_t head = task % heads;
            const nint_t batch = matrix / tiles;
            const nint_t tile = matrix % tiles;
            const nint_t begin = tile * RowsPerTask;
            const nint_t end = std::min(begin + RowsPerTask, sequence);
            const TileRows rows{end - begin};
            auto active_norm = normalized(batch, range(begin, end), reserve);
            auto bias_matrix =
                make_tensor(q_bias.data() + head * q_bias.stride(0),
                            make_layout(make_shape(rows, D),
                                        make_strides(cint<0>, cint<1>)));
            prepared_q(task_context, rows, D, C, active_norm,
                       packed_b(q_projection, head, D, C), bias_matrix,
                       q(batch, head, range(begin, end), reserve));
            prepared_projection(task_context, rows, D, C, active_norm,
                                packed_b(k_projection, head, D, C),
                                k(batch, head, range(begin, end), reserve));
            prepared_projection(task_context, rows, D, C, active_norm,
                                packed_b(v_projection, head, D, C),
                                v(batch, head, range(begin, end), reserve));
            prepared_projection(task_context, rows, D, C, active_norm,
                                packed_b(g_projection, head, D, C),
                                gate(batch, head, range(begin, end), reserve));
          }
        });
    projection_phase.close();
  } else {
    // Larger rows already expose several tiles. Keep normalization and all
    // head projections in one worker/session to avoid an additional team
    // launch and loss of normalized-tile locality.
    const nint_t projection_tasks = batch_count * tiles;
    const TileRows tile_rows{std::min(sequence, RowsPerTask)};
    auto sample_single = single_act(0, range(0, tile_rows), reserve);
    auto sample_norm = normalized(0, range(0, tile_rows), reserve);
    auto sample_q = q(0, 0, range(0, tile_rows), reserve);
    auto sample_k = k(0, 0, range(0, tile_rows), reserve);
    auto sample_q_bias =
        make_tensor(q_bias.data(), make_layout(make_shape(tile_rows, D),
                                               make_strides(cint<0>, cint<1>)));
    auto projection_phase = workspace.serial_scope("norm_and_projection");
    auto prepared_attention_norm = layer_norm.template prepare<Parallelism>(
        workspace, "attention_norm_scratch",
        tensor::unbind(tensor::input<ComputeType>(sample_single)),
        tensor::unbind(tensor::input<ComputeType>(attention_norm_scale)),
        tensor::unbind(tensor::input<ComputeType>(attention_norm_offset)),
        tensor::unbind(tensor::output<ComputeType>(sample_norm)));
    auto prepared_q = matmul_op.template prepare<Parallelism>(
        workspace, "q_projection_scratch", tile_rows, D, C,
        tensor::unbind(sample_norm),
        tensor::unbind(packed_b(q_projection, 0, D, C)),
        tensor::unbind(sample_q_bias), tensor::unbind(sample_q));
    auto prepared_projection = matmul_op.template prepare<Parallelism>(
        workspace, "projection_scratch", tile_rows, D, C,
        tensor::unbind(sample_norm),
        tensor::unbind(packed_b(k_projection, 0, D, C)),
        tensor::unbind(sample_k));
    workspace.parallel_for<Parallelism>(
        nint_t{0}, projection_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          const nint_t batch = task / tiles;
          const nint_t tile = task % tiles;
          const nint_t begin = tile * RowsPerTask;
          const nint_t end = std::min(begin + RowsPerTask, sequence);
          const TileRows rows{end - begin};
          auto active_norm = normalized(batch, range(begin, end), reserve);
          prepared_attention_norm(
              task_context, single_act(batch, range(begin, end), reserve),
              attention_norm_scale, attention_norm_offset, active_norm);
          for (nint_t head = 0; head < heads; ++head) {
            auto bias_matrix = make_tensor(
                q_bias.data() + head * q_bias.stride(0),
                make_layout(make_shape(rows, D),
                            make_strides(cint<0>, cint<1>)));
            prepared_q(task_context, rows, D, C, active_norm,
                       packed_b(q_projection, head, D, C), bias_matrix,
                       q(batch, head, range(begin, end), reserve));
            prepared_projection(task_context, rows, D, C, active_norm,
                                packed_b(k_projection, head, D, C),
                                k(batch, head, range(begin, end), reserve));
            prepared_projection(task_context, rows, D, C, active_norm,
                                packed_b(v_projection, head, D, C),
                                v(batch, head, range(begin, end), reserve));
            prepared_projection(task_context, rows, D, C, active_norm,
                                packed_b(g_projection, head, D, C),
                                gate(batch, head, range(begin, end), reserve));
          }
        });
    projection_phase.close();
  }

  // Attention, gated output projection, and transition block. No standalone
  // elementwise pass is needed: both gates are output transforms.
  const nint_t update_tasks = batch_count * tiles;
  const TileRows update_tile_rows{std::min(sequence, RowsPerTask)};
  const auto attention_shape = make_shape(update_tile_rows, D);
  const auto channel_shape = make_shape(update_tile_rows, C);
  const auto hidden_shape = make_shape(update_tile_rows, M);
  auto update_phase = workspace.serial_scope("attention_and_transition");
  auto attention_outputs =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "attention_output", attention_shape);
  auto transition_norms =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "transition_norm", channel_shape);
  auto transition_gates =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "transition_gate", hidden_shape);
  const nint_t update_rows = static_cast<nint_t>(update_tile_rows);
  auto sample_attention_output =
      attention_outputs(0, range(nint_t{0}, update_rows), reserve);
  auto sample_transition_norm =
      transition_norms(0, range(nint_t{0}, update_rows), reserve);
  auto sample_transition_gate =
      transition_gates(0, range(nint_t{0}, update_rows), reserve);
  auto sample_q = q(0, 0, range(nint_t{0}, update_rows), reserve);
  auto sample_k = k(0, 0, reserve, reserve);
  auto sample_v = v(0, 0, reserve, reserve);
  auto sample_gate = gate(0, 0, range(nint_t{0}, update_rows), reserve);
  auto sample_mask = key_padding_mask(0, reserve);
  auto sample_bias =
      pair_logits(0, 0, range(nint_t{0}, update_rows), reserve);
  auto gated_attention = tensor::output<ComputeType>(
      sample_attention_output, sigmoid_gate_transform(sample_gate));
  auto sample_single =
      single_act(0, range(nint_t{0}, update_rows), reserve);
  auto sample_output = output(0, range(nint_t{0}, update_rows), reserve);
  auto silu_gate =
      tensor::output<ComputeType>(sample_transition_gate, silu_transform());
  auto gated_value = tensor::output<ComputeType>(
      sample_transition_gate,
      transition_gate_transform(sample_transition_gate));
  const ComputeType attention_scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "attention_scratch", tensor::unbind(sample_q),
      tensor::unbind(sample_k), tensor::unbind(sample_v), tensor::nullopt,
      tensor::unbind(sample_mask), tensor::nullopt,
      tensor::unbind(sample_bias), tensor::unbind(gated_attention));
  auto prepared_attention_output = matmul_op.template prepare<Parallelism>(
      workspace, "attention_output_scratch", update_tile_rows, C, D,
      tensor::unbind(sample_attention_output),
      tensor::unbind(packed_b(o_projection, 0, C, D)),
      tensor::unbind(sample_single), tensor::unbind(sample_output));
  auto prepared_transition_norm = layer_norm.template prepare<Parallelism>(
      workspace, "transition_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_output)),
      tensor::unbind(tensor::input<ComputeType>(transition_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(transition_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_transition_norm)));
  auto prepared_transition_gate = matmul_op.template prepare<Parallelism>(
      workspace, "transition_gate_scratch", update_tile_rows, M, C,
      tensor::unbind(sample_transition_norm), tensor::unbind(transition1_gate),
      tensor::unbind(silu_gate));
  auto prepared_transition_value = matmul_op.template prepare<Parallelism>(
      workspace, "transition_value_scratch", update_tile_rows, M, C,
      tensor::unbind(sample_transition_norm), tensor::unbind(transition1_value),
      tensor::unbind(gated_value));
  auto prepared_transition_output = matmul_op.template prepare<Parallelism>(
      workspace, "transition_output_scratch", update_tile_rows, C, M,
      tensor::unbind(sample_transition_gate), tensor::unbind(transition2_b),
      tensor::unbind(sample_output), tensor::unbind(sample_output));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, update_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / tiles;
        const nint_t tile = task % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const TileRows rows{end - begin};
        auto active_single = single_act(batch, range(begin, end), reserve);
        auto active_output = output(batch, range(begin, end), reserve);
        auto attention_output = task_context.local(attention_outputs);
        auto transition_norm = task_context.local(transition_norms);
        auto transition_gate = task_context.local(transition_gates);
        auto active_attention = attention_output(range(0, rows), reserve);
        auto active_transition_norm =
            transition_norm(range(0, rows), reserve);
        auto active_transition_gate =
            transition_gate(range(0, rows), reserve);
        auto batch_mask = key_padding_mask(batch, reserve);

        for (nint_t head = 0; head < heads; ++head) {
          auto active_q = q(batch, head, range(begin, end), reserve);
          auto active_k = k(batch, head, reserve, reserve);
          auto active_v = v(batch, head, reserve, reserve);
          auto active_gate = gate(batch, head, range(begin, end), reserve);
          auto active_bias =
              pair_logits(batch, head, range(begin, end), reserve);
          prepared_attention(
              task_context, active_q, active_k, active_v, tensor::nullopt,
              batch_mask, tensor::nullopt, active_bias,
              tensor::output<ComputeType>(
                  active_attention, sigmoid_gate_transform(active_gate)),
              attention_scale);
          if (head == 0) {
            prepared_attention_output(
                task_context, rows, C, D, active_attention,
                packed_b(o_projection, head, C, D), active_single,
                active_output);
          } else {
            prepared_attention_output(
                task_context, rows, C, D, active_attention,
                packed_b(o_projection, head, C, D), active_output,
                active_output);
          }
        }

        prepared_transition_norm(
            task_context, active_output, transition_norm_scale,
            transition_norm_offset, active_transition_norm);
        prepared_transition_gate(
            task_context, rows, M, C, active_transition_norm, transition1_gate,
            tensor::output<ComputeType>(active_transition_gate,
                                        silu_transform()));
        prepared_transition_value(
            task_context, rows, M, C, active_transition_norm,
            transition1_value,
            tensor::output<ComputeType>(
                active_transition_gate,
                transition_gate_transform(active_transition_gate)));
        prepared_transition_output(task_context, rows, C, M,
                                   active_transition_gate, transition2_b,
                                   active_output, active_output);
      });
  update_phase.close();
}
