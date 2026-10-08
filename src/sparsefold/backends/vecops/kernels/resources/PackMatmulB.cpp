// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file PackMatmulB.cpp @brief Persistently pack BF16 matmul B operands. */

#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
using NExtent = spec::N;
using KExtent = spec::K;
using PExtent = spec::P;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "PackMatmulB requires an AMX or SME BF16 matmul target"
#endif

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto input,
                TensorLike auto output) {
  const NExtent n{input.size(1)};
  const KExtent k{input.size(2)};
  const PExtent p{output.size(1)};
  assert_shape(input, {vecops::any, n, k},
               {vecops::any, k, cint<1>});
  assert_shape(output, {vecops::any, p}, {p, cint<1>});
  const nint_t batch = input.size(0);
  if (batch == 0)
    return;

  const auto logical_layout = make_layout(make_shape(n, k));
  const auto packed_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  const nint_t packed_elements = numel(packed_layout);
  VECOPS_ASSERT(static_cast<nint_t>(p) >= packed_elements,
                "packed output is too small");
  auto packer = ops::matmul_pack(
      ops::MatmulPackConfig<MatmulAtom, matmul::Operand::B>{});
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  auto packed_pattern = make_tensor(output.data(), packed_layout);
  auto phase = workspace.serial_scope("pack_matmul_b");
  auto prepared_packer = packer.template prepare<Parallelism>(
      workspace, "packer_scratch",
      tensor::unbind(input(0, reserve, reserve)),
      tensor::unbind(packed_pattern));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, batch, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t index = work_item.begin();
        auto packed = make_tensor(
            output.data() + index * output.stride(0), packed_layout);
        prepared_packer(task_context, input(index, reserve, reserve), packed);
      });
  phase.close();
}
