// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file SingleDistributedUpdate.cpp
 *  @brief Post-collective Evoformer single attention and transition.
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

using LocalExtent = spec::L;
constexpr auto N = spec::N;
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
#error "Distributed single update requires AMX or SME"
#endif

namespace {

constexpr nint_t RowsPerTask = 32;
using TileRows = meta::Dynamic<1, 1, RowsPerTask>;
using AttentionMatmulConfig = ops::MatmulConfig<
    MatmulAtom,
    matmul::family_selection::Require<matmul::kernel_family::GenericTiled>>;
using AttentionConfig = ops::AttentionConfig<
    AttentionMatmulConfig, 32, 32, ops::AttentionCausalMode::none,
    ops::SDPAStrategy::automatic, vec::Accuracy::Estimate>;

template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, nint_t index, Spatial spatial,
              Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data() + index * packed.stride(0), layout);
}

template <TensorLike Gate> auto sigmoid_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            const auto coordinate = context.lane_coord(0);
            const auto gate_value = vec::load_convert(
                tag, gate_data + coordinate[0] * gate_stride + coordinate[1],
                context.active_option(tag), vec::opt::zero);
            return vec::mul(
                tag, value,
                sparsefold_vecops::stable_sigmoid<ComputeType>(tag, gate_value));
          });
}

auto silu_transform() {
  return make_elementwise_vec_transform<ComputeType, ComputeType>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(
            tag, value,
            sparsefold_vecops::stable_sigmoid<ComputeType>(tag, value));
      });
}

template <TensorLike Gate> auto transition_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            const auto coordinate = context.lane_coord(0);
            const auto old_gate = vec::load_convert(
                tag, gate_data + coordinate[0] * gate_stride + coordinate[1],
                context.active_option(tag), vec::opt::zero);
            return vec::mul(tag, old_gate, value);
          });
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace,
                TensorLike auto single_act, TensorLike auto key_padding_mask,
                TensorLike auto pair_logits, TensorLike auto q,
                TensorLike auto global_k, TensorLike auto global_v,
                TensorLike auto gate, TensorLike auto o_projection,
                TensorLike auto transition_norm_scale,
                TensorLike auto transition_norm_offset,
                TensorLike auto transition1, TensorLike auto transition2,
                TensorLike auto output) {
  assert_shape(single_act, {vecops::any, vecops::any, C},
               {vecops::any, C, cint<1>});
  assert_shape(key_padding_mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(pair_logits, {vecops::any, H, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(q, {vecops::any, H, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(global_k, {vecops::any, H, N, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(global_v, {vecops::any, H, N, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(gate, {vecops::any, H, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(o_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(transition_norm_scale, {C}, {cint<1>});
  assert_shape(transition_norm_offset, {C}, {cint<1>});
  assert_shape(transition1, {cint<2>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(transition2, {cint<1>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, vecops::any, C},
               {vecops::any, C, cint<1>});

  const nint_t batch_count = single_act.size(0);
  const LocalExtent local_extent{single_act.size(1)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  if (batch_count == 0 || local_rows == 0)
    return;

  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto attention_op = ops::scaled_dot_product_attention(AttentionConfig{});
  auto transition_gate = packed_b(transition1, 0, M, C);
  auto transition_value = packed_b(transition1, 1, M, C);
  auto transition_output = packed_b(transition2, 0, C, M);

  const nint_t tiles = ceil_div(local_rows, RowsPerTask);
  const nint_t tasks = batch_count * tiles;
  const TileRows sample_rows{std::min(local_rows, RowsPerTask)};
  const auto attention_shape = make_shape(sample_rows, D);
  const auto channel_shape = make_shape(sample_rows, C);
  const auto hidden_shape = make_shape(sample_rows, M);
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  auto phase = workspace.serial_scope("single_distributed_update");
  auto attention_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("attention",
                                                               attention_shape);
  auto norm_storage = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "norm", channel_shape);
  auto hidden_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("hidden",
                                                               hidden_shape);
  const execution::TaskContext<Parallelism> sample_task{0};
  auto sample_attention_worker = sample_task.local(attention_storage);
  auto sample_norm_worker = sample_task.local(norm_storage);
  auto sample_hidden_worker = sample_task.local(hidden_storage);
  auto sample_attention =
      sample_attention_worker(range(0, sample_rows), reserve);
  auto sample_norm = sample_norm_worker(range(0, sample_rows), reserve);
  auto sample_hidden = sample_hidden_worker(range(0, sample_rows), reserve);
  auto sample_single = single_act(0, range(0, sample_rows), reserve);
  auto sample_output = output(0, range(0, sample_rows), reserve);
  const ComputeType scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "attention_scratch",
      tensor::unbind(q(0, 0, range(0, sample_rows), reserve)),
      tensor::unbind(global_k(0, 0, reserve, reserve)),
      tensor::unbind(global_v(0, 0, reserve, reserve)), tensor::nullopt,
      tensor::unbind(key_padding_mask(0, reserve)), tensor::nullopt,
      tensor::unbind(pair_logits(0, 0, range(0, sample_rows), reserve)),
      tensor::unbind(tensor::output<ComputeType>(
          sample_attention,
          sigmoid_gate_transform(gate(0, 0, range(0, sample_rows), reserve)))));
  auto prepared_attention_output = matmul_op.template prepare<Parallelism>(
      workspace, "attention_output_scratch", sample_rows, C, D,
      tensor::unbind(sample_attention),
      tensor::unbind(packed_b(o_projection, 0, C, D)),
      tensor::unbind(sample_single), tensor::unbind(sample_output));
  auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
      workspace, "transition_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_output)),
      tensor::unbind(tensor::input<ComputeType>(transition_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(transition_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_norm)));
  auto prepared_transition_gate = matmul_op.template prepare<Parallelism>(
      workspace, "transition_gate_scratch", sample_rows, M, C,
      tensor::unbind(sample_norm), tensor::unbind(transition_gate),
      tensor::unbind(
          tensor::output<ComputeType>(sample_hidden, silu_transform())));
  auto prepared_transition_value = matmul_op.template prepare<Parallelism>(
      workspace, "transition_value_scratch", sample_rows, M, C,
      tensor::unbind(sample_norm), tensor::unbind(transition_value),
      tensor::unbind(tensor::output<ComputeType>(
          sample_hidden, transition_gate_transform(sample_hidden))));
  auto prepared_transition_output = matmul_op.template prepare<Parallelism>(
      workspace, "transition_output_scratch", sample_rows, C, M,
      tensor::unbind(sample_hidden), tensor::unbind(transition_output),
      tensor::unbind(sample_output), tensor::unbind(sample_output));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / tiles;
        const nint_t tile = task % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, local_rows);
        const TileRows rows{end - begin};
        auto attention = task_context.local(attention_storage);
        auto norm = task_context.local(norm_storage);
        auto hidden = task_context.local(hidden_storage);
        auto active_single = single_act(batch, range(begin, end), reserve);
        auto active_output = output(batch, range(begin, end), reserve);
        auto active_attention = attention(range(0, rows), reserve);
        auto active_norm = norm(range(0, rows), reserve);
        auto active_hidden = hidden(range(0, rows), reserve);
        for (nint_t head = 0; head < heads; ++head) {
          prepared_attention(
              task_context, q(batch, head, range(begin, end), reserve),
              global_k(batch, head, reserve, reserve),
              global_v(batch, head, reserve, reserve), tensor::nullopt,
              key_padding_mask(batch, reserve), tensor::nullopt,
              pair_logits(batch, head, range(begin, end), reserve),
              tensor::output<ComputeType>(
                  active_attention,
                  sigmoid_gate_transform(
                      gate(batch, head, range(begin, end), reserve))),
              scale);
          if (head == 0) {
            prepared_attention_output(task_context, rows, C, D,
                                      active_attention,
                                      packed_b(o_projection, head, C, D),
                                      active_single, active_output);
          } else {
            prepared_attention_output(task_context, rows, C, D,
                                      active_attention,
                                      packed_b(o_projection, head, C, D),
                                      active_output, active_output);
          }
        }
        prepared_layer_norm(task_context, active_output, transition_norm_scale,
                            transition_norm_offset, active_norm);
        prepared_transition_gate(
            task_context, rows, M, C, active_norm, transition_gate,
            tensor::output<ComputeType>(active_hidden, silu_transform()));
        prepared_transition_value(
            task_context, rows, M, C, active_norm, transition_value,
            tensor::output<ComputeType>(
                active_hidden, transition_gate_transform(active_hidden)));
        prepared_transition_output(task_context, rows, C, M, active_hidden,
                                   transition_output, active_output,
                                   active_output);
      });
  phase.close();
}
