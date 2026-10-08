// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file TransitionBlock.cpp @brief Fused AlphaFold 3 transition block. */

#include <algorithm>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto M = spec::M;
using B = spec::B;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "TransitionBlock requires an AMX or SME BF16 matmul target"
#endif

namespace {

constexpr nint_t RowsPerTask = 128;

template <TensorLike Gate>
auto transition1_transform(Gate gate, bool multiply_gate) {
  auto *gate_data = gate.data();
  const nint_t row_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, row_stride, multiply_gate](
          auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        if (multiply_gate) {
          VECOPS_ASSERT(context.vector_axis == 1,
                        "gate epilogue must traverse columns");
          const auto coordinate = context.lane_coord(0);
          const nint_t row = coordinate[0];
          const nint_t column = coordinate[1];
          const auto old_gate = vec::load_convert(
              tag, gate_data + row * row_stride + column,
              context.active_option(tag), vec::opt::zero);
          return vec::mul(tag, old_gate, value);
        }
#if defined(HAS_SME)
        const auto sigmoid =
            sparsefold_vecops::polynomial_sigmoid<ComputeType>(tag, value);
#else
        const auto sigmoid =
            sparsefold_vecops::stable_sigmoid<ComputeType>(tag, value);
#endif
        return vec::mul(tag, value, sigmoid);
      });
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto ln_scale, TensorLike auto ln_offset,
                TensorLike auto transition1, TensorLike auto transition2,
                TensorLike auto output) {
  assert_shape(act, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(ln_scale, {C}, {cint<1>});
  assert_shape(ln_offset, {C}, {cint<1>});
  assert_shape(transition1, {C, cint<2>, M}, {vecops::any, M, cint<1>});
  assert_shape(transition2, {M, C}, {C, cint<1>});
  assert_shape(output, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  const auto batch = B{act.size(0)};
  const auto rows = batch * N;
  const auto row_count = static_cast<nint_t>(rows);
  const auto row_shape = make_shape(rows, C);
  auto act_rows = make_tensor(act.data(), row_shape);
  auto output_rows = make_tensor(output.data(), row_shape);

  auto transition1_gate = transpose<0, 1>(transition1(reserve, 0, reserve));
  auto transition1_value = transpose<0, 1>(transition1(reserve, 1, reserve));
  auto transition2_transposed = transpose<0, 1>(transition2);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});

  if (row_count == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  // The worker buffers and operator scratch overlap for the full parallel
  // region. Request all replicas before entering the team so trace/replay sees
  // a deterministic allocation tree independent of worker arrival order.
  const nint_t workspace_rows = std::min(row_count, RowsPerTask);
  const auto workspace_row_shape = make_shape(workspace_rows, C);
  const auto workspace_hidden_shape = make_shape(workspace_rows, M);
  auto phase = workspace.serial_scope("transition_block");
  auto normalized_workers =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "normalized", workspace_row_shape);
  auto gate_workers = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "gate", workspace_hidden_shape);

  auto normalized_pattern = normalized_workers(0, ellipsis);
  auto gate_pattern = gate_workers(0, ellipsis);
  auto act_pattern = act_rows(range(0, workspace_rows), reserve);
  auto output_pattern = output_rows(range(0, workspace_rows), reserve);

  auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(act_pattern)),
      tensor::unbind(tensor::input<ComputeType>(ln_scale)),
      tensor::unbind(tensor::input<ComputeType>(ln_offset)),
      tensor::unbind(tensor::output<ComputeType>(normalized_pattern)));
  auto prepared_transition1 = matmul_op.template prepare<Parallelism>(
      workspace, "transition1_scratch", workspace_rows, M, C,
      tensor::unbind(normalized_pattern), tensor::unbind(transition1_gate),
      tensor::unbind(tensor::output<ComputeType>(
          gate_pattern, transition1_transform(gate_pattern, false))));
  auto prepared_transition2 = matmul_op.template prepare<Parallelism>(
      workspace, "transition2_scratch", workspace_rows, C, M,
      tensor::unbind(gate_pattern), tensor::unbind(transition2_transposed),
      tensor::unbind(act_pattern), tensor::unbind(output_pattern));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, row_count, cint<RowsPerTask>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t row_begin = work_item.begin();
        const nint_t row_end = work_item.end();
        const nint_t active_rows = row_end - row_begin;
        auto active_act = act_rows(range(row_begin, row_end), reserve);
        auto active_output = output_rows(range(row_begin, row_end), reserve);
        auto normalized_storage = task_context.local(normalized_workers);
        auto gate_storage = task_context.local(gate_workers);
        auto normalized =
            normalized_storage(range(0, active_rows), reserve);
        auto gate = gate_storage(range(0, active_rows), reserve);

        prepared_layer_norm(task_context, active_act, ln_scale, ln_offset,
                            normalized);
        prepared_transition1(
            task_context, active_rows, M, C, normalized, transition1_gate,
            tensor::output<ComputeType>(
                gate, transition1_transform(gate, false)));
        prepared_transition1(
            task_context, active_rows, M, C, normalized, transition1_value,
            tensor::output<ComputeType>(
                gate, transition1_transform(gate, true)));
        prepared_transition2(task_context, active_rows, C, M, gate,
                             transition2_transposed, active_act,
                             active_output);
      });
  phase.close();
}
