// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file OuterDistributedLeftProjection.cpp
 *  @brief Local work overlapped with the outer-product right gather.
 */

#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
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

void __kernel__(execution::WorkspaceContext &workspace,
                TensorLike auto normalized, TensorLike auto local_mask,
                TensorLike auto left_projection, TensorLike auto left) {
  assert_shape(normalized, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(local_mask, {vecops::any, M, vecops::any},
               {vecops::any, vecops::any, cint<1>});
  assert_shape(left, {vecops::any, M, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  const nint_t batch_count = normalized.size(0);
  const LocalExtent local_extent{normalized.size(2)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t msa_rows = static_cast<nint_t>(M);
  if (batch_count == 0 || local_rows == 0)
    return;

  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto left_b = packed_b(left_projection);
  const nint_t tasks = batch_count * msa_rows;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  auto phase = workspace.serial_scope("outer_distributed_left_projection");
  auto prepared_matmul = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", local_extent, D, C,
      tensor::unbind(normalized(0, 0, reserve, reserve)),
      tensor::unbind(left_b),
      tensor::unbind(tensor::output<ComputeType>(
          left(0, 0, reserve, reserve),
          mask_transform(local_mask(0, 0, reserve)))));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / msa_rows;
        const nint_t msa = task % msa_rows;
        prepared_matmul(
            task_context, normalized(batch, msa, reserve, reserve), left_b,
            tensor::output<ComputeType>(
                left(batch, msa, reserve, reserve),
                mask_transform(local_mask(batch, msa, reserve))));
      });
  phase.close();
}
