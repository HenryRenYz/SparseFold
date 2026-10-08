// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file OuterDistributedRightProjection.cpp
 *  @brief Send-first phase for distributed outer-product mean.
 */

#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using LocalExtent = spec::L;
constexpr auto M = spec::M;
constexpr auto C = spec::C;
constexpr auto D = spec::OuterChannels;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed outer projection requires AMX or SME"
#endif

namespace {
template <TensorLike Packed> auto packed_b(Packed packed) {
  const auto layout = matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
      make_layout(make_shape(D, C)));
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data(), layout);
}

template <TensorLike Mask> auto mask_transform(Mask mask) {
  const auto *mask_data = mask.data();
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [mask_data](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const auto scale = vec::fill(
            tag, static_cast<ComputeType>(
                     static_cast<unsigned>(mask_data[coordinate[0]])));
        return vec::mul(tag, value, scale);
      });
}
} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto local_mask, TensorLike auto norm_scale,
                TensorLike auto norm_offset, TensorLike auto right_projection,
                TensorLike auto normalized, TensorLike auto right_send) {
  assert_shape(act, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(local_mask, {vecops::any, M, vecops::any},
               {vecops::any, vecops::any, cint<1>});
  assert_shape(normalized, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(right_send, {vecops::any, vecops::any, M, D},
               {vecops::any, vecops::any, D, cint<1>});
  const nint_t batch_count = act.size(0);
  const LocalExtent local_extent{act.size(2)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t msa_rows = static_cast<nint_t>(M);
  if (batch_count == 0 || local_rows == 0)
    return;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto right_b = packed_b(right_projection);
  const nint_t tasks = batch_count * msa_rows;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  auto phase = workspace.serial_scope("outer_distributed_right_projection");
  auto norm_pattern = tensor::unbind(
      normalized(0, 0, reserve, reserve));
  auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(
          act(0, 0, reserve, reserve))),
      tensor::unbind(tensor::input<ComputeType>(norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(
          normalized(0, 0, reserve, reserve))));
  auto prepared_matmul = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", local_extent, D, C,
      std::move(norm_pattern), tensor::unbind(right_b),
      tensor::unbind(tensor::output<ComputeType>(
          right_send(reserve, 0, 0, reserve),
          mask_transform(local_mask(0, 0, reserve)))));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / msa_rows;
        const nint_t msa = task % msa_rows;
        auto active_norm = normalized(batch, msa, reserve, reserve);
        auto active_mask = local_mask(batch, msa, reserve);
        prepared_layer_norm(
            task_context, act(batch, msa, reserve, reserve), norm_scale,
            norm_offset, active_norm);
        prepared_matmul(
            task_context, active_norm, right_b,
            tensor::output<ComputeType>(
                right_send(reserve, batch, msa, reserve),
                mask_transform(active_mask)));
      });
  phase.close();
}
