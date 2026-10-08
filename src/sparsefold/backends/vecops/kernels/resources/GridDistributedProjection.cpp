// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file GridDistributedProjection.cpp
 *  @brief Local normalization and pair-bias projection for distributed grid attention.
 */

#include <type_traits>

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
constexpr auto C = spec::C;
constexpr auto H = spec::H;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed grid projection requires AMX or SME"
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

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto norm_scale, TensorLike auto norm_offset,
                TensorLike auto bias_projection, TensorLike auto normalized,
                TensorLike auto bias_send) {
  assert_shape(act, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(norm_scale, {C}, {cint<1>});
  assert_shape(norm_offset, {C}, {cint<1>});
  assert_shape(bias_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(normalized, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(bias_send, {vecops::any, vecops::any, H, N},
               {vecops::any, vecops::any, N, cint<1>});

  const nint_t batch_count = act.size(0);
  const LocalExtent local_extent{act.size(1)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  if (batch_count == 0 || local_rows == 0)
    return;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto bias_b = packed_b(bias_projection, H, C);
  const nint_t tasks = batch_count * local_rows;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  auto phase = workspace.serial_scope("grid_distributed_projection");

  auto norm_input_pattern = tensor::unbind(tensor::input<ComputeType>(
      act(0, 0, reserve, reserve)));
  auto norm_scale_pattern = tensor::unbind(tensor::input<ComputeType>(
      norm_scale));
  auto norm_offset_pattern = tensor::unbind(tensor::input<ComputeType>(
      norm_offset));
  auto norm_output_pattern = tensor::unbind(tensor::output<ComputeType>(
      normalized(0, 0, reserve, reserve)));
  auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch", std::move(norm_input_pattern),
      std::move(norm_scale_pattern), std::move(norm_offset_pattern),
      std::move(norm_output_pattern));

  auto matmul_input_pattern = tensor::unbind(
      normalized(0, 0, reserve, reserve));
  auto matmul_weight_pattern = tensor::unbind(bias_b);
  auto matmul_output_pattern = tensor::unbind(transpose<0, 1>(
      bias_send(0, 0, reserve, reserve)));
  auto prepared_matmul = matmul_op.template prepare<Parallelism>(
      workspace, "bias_projection_scratch", N, H, C,
      std::move(matmul_input_pattern), std::move(matmul_weight_pattern),
      std::move(matmul_output_pattern));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / local_rows;
        const nint_t row = task % local_rows;
        auto active_norm = normalized(batch, row, reserve, reserve);
        prepared_layer_norm(
            task_context, act(batch, row, reserve, reserve), norm_scale,
            norm_offset, active_norm);
        prepared_matmul(
            task_context, active_norm, bias_b,
            transpose<0, 1>(bias_send(row, batch, reserve, reserve)));
      });
  phase.close();
}
