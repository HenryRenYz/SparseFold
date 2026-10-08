// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file SingleDistributedProjection.cpp
 *  @brief Local projection phase for distributed Evoformer single update.
 */

#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

using LocalExtent = spec::L;
constexpr auto N = spec::N;
constexpr auto Q = spec::Q;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::D;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed single projection requires AMX or SME"
#endif

namespace {

constexpr nint_t RowsPerTask = 32;
using TileRows = meta::Dynamic<1, 1, RowsPerTask>;

template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, nint_t index, Spatial spatial,
              Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data() + index * packed.stride(0), layout);
}

} // namespace

void __kernel__(
    execution::WorkspaceContext &workspace, TensorLike auto pair_act,
    TensorLike auto single_act, TensorLike auto pair_norm_scale,
    TensorLike auto pair_norm_offset, TensorLike auto logits_projection,
    TensorLike auto attention_norm_scale,
    TensorLike auto attention_norm_offset, TensorLike auto q_projection,
    TensorLike auto q_bias, TensorLike auto k_projection,
    TensorLike auto v_projection, TensorLike auto g_projection,
    TensorLike auto pair_logits, TensorLike auto q, TensorLike auto k,
    TensorLike auto v, TensorLike auto gate) {
  assert_shape(pair_act, {vecops::any, vecops::any, N, Q},
               {vecops::any, vecops::any, Q, cint<1>});
  assert_shape(single_act, {vecops::any, vecops::any, C},
               {vecops::any, C, cint<1>});
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
  assert_shape(pair_logits, {vecops::any, H, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(q, {vecops::any, H, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(k, {vecops::any, H, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(v, {vecops::any, H, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(gate, {vecops::any, H, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});

  const nint_t batch_count = pair_act.size(0);
  const LocalExtent local_extent{pair_act.size(1)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t heads = static_cast<nint_t>(H);
  if (batch_count == 0 || local_rows == 0)
    return;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto logits_b = packed_b(logits_projection, 0, H, Q);
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const nint_t pair_tasks = batch_count * local_rows;
  const auto pair_norm_shape = make_shape(N, Q);
  auto pair_phase = workspace.serial_scope("pair_logits");
  auto pair_norm_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("normalized", pair_norm_shape);
  auto pair_norm_pattern = tensor::unbind(
      pair_norm_workers(0, reserve, reserve));
  auto prepared_pair_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(
          pair_act(0, 0, reserve, reserve))),
      tensor::unbind(tensor::input<ComputeType>(pair_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(pair_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(
          pair_norm_workers(0, reserve, reserve))));
  auto prepared_logits = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", N, H, Q,
      std::move(pair_norm_pattern), tensor::unbind(logits_b),
      tensor::unbind(transpose<0, 1>(
          pair_logits(0, reserve, 0, reserve))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, pair_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / local_rows;
        const nint_t row = task % local_rows;
        auto pair_norm = task_context.local(pair_norm_workers);
        prepared_pair_norm(
            task_context, pair_act(batch, row, reserve, reserve),
            pair_norm_scale, pair_norm_offset, pair_norm);
        prepared_logits(
            task_context, pair_norm, logits_b,
            transpose<0, 1>(
                pair_logits(batch, reserve, row, reserve)));
      });
  pair_phase.close();

  const nint_t tiles = ceil_div(local_rows, RowsPerTask);
  const nint_t projection_tasks = batch_count * tiles;
  const TileRows sample_rows{std::min(local_rows, RowsPerTask)};
  const auto norm_shape = make_shape(sample_rows, C);
  auto projection_phase = workspace.serial_scope("single_projection");
  auto norm_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("normalized", norm_shape);
  auto sample_norm = norm_workers(0, reserve, reserve);
  auto sample_q = q(0, 0, range(0, sample_rows), reserve);
  auto sample_bias = make_tensor(
      q_bias.data(), make_layout(make_shape(sample_rows, D),
                                 make_strides(cint<0>, cint<1>)));
  auto prepared_attention_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(
          single_act(0, range(0, sample_rows), reserve))),
      tensor::unbind(tensor::input<ComputeType>(attention_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(attention_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_norm)));
  auto prepared_q = matmul_op.template prepare<Parallelism>(
      workspace, "q_projection_scratch", sample_rows, D, C,
      tensor::unbind(sample_norm),
      tensor::unbind(packed_b(q_projection, 0, D, C)),
      tensor::unbind(tensor::input<ComputeType>(sample_bias)),
      tensor::unbind(tensor::output<ComputeType>(sample_q)));
  auto prepared_projection = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", sample_rows, D, C,
      tensor::unbind(sample_norm),
      tensor::unbind(packed_b(k_projection, 0, D, C)),
      tensor::unbind(k(0, 0, range(0, sample_rows), reserve)));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, projection_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / tiles;
        const nint_t tile = task % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, local_rows);
        const TileRows rows{end - begin};
        auto normalized = task_context.local(norm_workers);
        auto active_norm = normalized(range(0, rows), reserve);
        prepared_attention_norm(
            task_context, single_act(batch, range(begin, end), reserve),
            attention_norm_scale, attention_norm_offset, active_norm);
        for (nint_t head = 0; head < heads; ++head) {
          auto bias = make_tensor(
              q_bias.data() + head * q_bias.stride(0),
              make_layout(make_shape(rows, D),
                          make_strides(cint<0>, cint<1>)));
          prepared_q(
              task_context, active_norm,
              packed_b(q_projection, head, D, C),
              tensor::input<ComputeType>(bias),
              tensor::output<ComputeType>(
                  q(batch, head, range(begin, end), reserve)));
          prepared_projection(
              task_context, active_norm,
              packed_b(k_projection, head, D, C),
              k(batch, head, range(begin, end), reserve));
          prepared_projection(
              task_context, active_norm,
              packed_b(v_projection, head, D, C),
              v(batch, head, range(begin, end), reserve));
          prepared_projection(
              task_context, active_norm,
              packed_b(g_projection, head, D, C),
              gate(batch, head, range(begin, end), reserve));
        }
      });
  projection_phase.close();
}
