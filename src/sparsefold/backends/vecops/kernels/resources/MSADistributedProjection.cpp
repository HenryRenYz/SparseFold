// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file MSADistributedProjection.cpp
 *  @brief Local projection phase for distributed MSA attention.
 */

#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/Softmax.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

using LocalExtent = spec::L;
constexpr auto M = spec::M;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto Q = spec::Q;
constexpr auto H = spec::H;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed MSA projection requires AMX or SME"
#endif

namespace {
template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, Spatial spatial, Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data(), layout);
}
} // namespace

void __kernel__(
    execution::WorkspaceContext &workspace, TensorLike auto act,
    TensorLike auto mask, TensorLike auto pair_act,
    TensorLike auto act_norm_scale, TensorLike auto act_norm_offset,
    TensorLike auto pair_norm_scale, TensorLike auto pair_norm_offset,
    TensorLike auto pair_logits_projection, TensorLike auto value_projection,
    TensorLike auto gate_projection, TensorLike auto weights,
    TensorLike auto value_send, TensorLike auto gate) {
  assert_shape(act, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(mask, {vecops::any, M, N}, {vecops::any, N, cint<1>});
  assert_shape(pair_act, {vecops::any, vecops::any, N, Q},
               {vecops::any, vecops::any, Q, cint<1>});
  assert_shape(weights, {vecops::any, H, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(value_send, {vecops::any, vecops::any, M, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(gate, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});

  const nint_t batch_count = act.size(0);
  const LocalExtent local_extent{act.size(2)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t msa_rows = static_cast<nint_t>(M);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t heads = static_cast<nint_t>(H);
  if (batch_count == 0 || local_rows == 0)
    return;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto softmax_op = ops::softmax(
      ops::SoftmaxConfig<ComputeType, vec::Accuracy::Estimate>{});
  auto pair_b = packed_b(pair_logits_projection, H, Q);
  auto value_b = packed_b(value_projection, C, C);
  auto gate_b = packed_b(gate_projection, C, C);
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const nint_t pair_tasks = batch_count * local_rows;
  const auto pair_norm_shape = make_shape(N, Q);
  const auto logits_shape = make_shape(N, H);
  auto pair_phase = workspace.serial_scope("msa_pair_weights");
  auto pair_norm_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("normalized", pair_norm_shape);
  auto logits_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("logits", logits_shape);
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
      tensor::unbind(pair_norm_workers(0, reserve, reserve)),
      tensor::unbind(pair_b),
      tensor::unbind(logits_workers(0, reserve, reserve)));
  auto prepared_softmax = softmax_op.template prepare<Parallelism>(
      workspace, "softmax_scratch",
      tensor::unbind(tensor::input<ComputeType>(
          weights(0, 0, 0, reserve))),
      tensor::unbind(tensor::output<ComputeType>(
          weights(0, 0, 0, reserve))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, pair_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / local_rows;
        const nint_t query = task % local_rows;
        auto pair_norm = task_context.local(pair_norm_workers);
        auto logits = task_context.local(logits_workers);
        prepared_pair_norm(
            task_context, pair_act(batch, query, reserve, reserve),
            pair_norm_scale, pair_norm_offset, pair_norm);
        prepared_logits(task_context, pair_norm, pair_b, logits);
        for (nint_t head = 0; head < heads; ++head) {
          auto active_weight = weights(batch, head, query, reserve);
          for (nint_t key = 0; key < sequence; ++key) {
            bool valid = false;
            for (nint_t msa = 0; msa < msa_rows; ++msa)
              valid = valid || static_cast<bool>(mask(batch, msa, key));
            const ComputeType bias = valid ? static_cast<ComputeType>(0)
                                           : static_cast<ComputeType>(-1e9);
            active_weight(key) = vecops::bfloat16_t(
                static_cast<ComputeType>(logits(key, head)) + bias);
          }
          prepared_softmax(
              task_context, active_weight, active_weight);
        }
      });
  pair_phase.close();

  const nint_t act_tasks = batch_count * msa_rows;
  const auto normalized_shape = make_shape(LocalExtent{local_rows}, C);
  auto act_phase = workspace.serial_scope("msa_value_gate");
  auto norm_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("normalized", normalized_shape);
  auto norm_pattern = tensor::unbind(norm_workers(0, reserve, reserve));
  auto prepared_act_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(
          act(0, 0, reserve, reserve))),
      tensor::unbind(tensor::input<ComputeType>(act_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(act_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(
          norm_workers(0, reserve, reserve))));
  auto prepared_value = matmul_op.template prepare<Parallelism>(
      workspace, "value_projection_scratch", LocalExtent{local_rows}, C, C,
      std::move(norm_pattern), tensor::unbind(value_b),
      tensor::unbind(value_send(reserve, 0, 0, reserve)));
  auto prepared_gate = matmul_op.template prepare<Parallelism>(
      workspace, "gate_projection_scratch", LocalExtent{local_rows}, C, C,
      tensor::unbind(norm_workers(0, reserve, reserve)),
      tensor::unbind(gate_b),
      tensor::unbind(gate(0, 0, reserve, reserve)));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, act_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / msa_rows;
        const nint_t msa = task % msa_rows;
        auto active_norm = task_context.local(norm_workers);
        prepared_act_norm(
            task_context, act(batch, msa, reserve, reserve),
            act_norm_scale, act_norm_offset, active_norm);
        prepared_value(
            task_context, active_norm, value_b,
            value_send(reserve, batch, msa, reserve));
        prepared_gate(
            task_context, active_norm, gate_b,
            gate(batch, msa, reserve, reserve));
      });
  act_phase.close();
}
