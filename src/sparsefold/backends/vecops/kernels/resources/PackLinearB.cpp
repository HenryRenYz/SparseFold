// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file PackLinearB.cpp @brief Pack generic Dynamic-shape Linear B operands. */

#include <concepts>
#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

using NExtent = spec::N;
using KExtent = spec::K;
using PExtent = spec::P;
constexpr auto FP32MatmulPrecision = spec::FP32MatmulPrecision;

#if defined(ARCH_X86_FAMILY)
template <typename PackedType> struct SelectMatmulAtom {
  static_assert(std::same_as<PackedType, vecops::bfloat16_t> ||
                    std::same_as<PackedType, vecops::float16_t>,
                "AMX has no FP32 matmul atom; FP32 Linear requires "
                "fp32_matmul_precision=medium");
  using type = std::conditional_t<std::same_as<PackedType, vecops::bfloat16_t>,
                                  matmul::AMX_BF16F32,
                                  matmul::AMX_F16F32>;
};
#elif defined(HAS_SME)
template <typename PackedType> struct SelectMatmulAtom {
  static_assert(std::same_as<PackedType, vecops::bfloat16_t> ||
                    std::same_as<PackedType, vecops::float16_t> ||
                    std::same_as<PackedType, vecops::float32_t>,
                "SME Linear only supports BF16, FP16 and FP32 weights");
  using type = std::conditional_t<
      std::same_as<PackedType, vecops::bfloat16_t>, matmul::SME_BF16F32,
      std::conditional_t<std::same_as<PackedType, vecops::float16_t>,
                         matmul::SME_F16F32, matmul::SME_F32F32>>;
};
#else
#error "PackLinearB requires an AMX or SME matmul target"
#endif

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto input,
                TensorLike auto output) {
  const NExtent n{input.size(1)};
  const KExtent k{input.size(2)};
  const PExtent p{output.size(1)};
  assert_shape(input, {vecops::any, n, k},
               {vecops::any, vecops::any, cint<1>});
  assert_shape(output, {vecops::any, p}, {p, cint<1>});
  const nint_t batch = input.size(0);
  if (batch == 0)
    return;

  using InputType = std::remove_const_t<
      typename std::remove_cvref_t<decltype(input)>::ElementType>;
  using PackedType = std::remove_const_t<
      typename std::remove_cvref_t<decltype(output)>::ElementType>;
  static_assert(std::same_as<InputType, vecops::bfloat16_t> ||
                    std::same_as<InputType, vecops::float16_t> ||
                    std::same_as<InputType, vecops::float32_t>,
                "PackLinearB only supports BF16, FP16 and FP32 inputs");
  static_assert(!std::same_as<InputType, vecops::float32_t> ||
                    ((FP32MatmulPrecision == 0 &&
                      std::same_as<PackedType, vecops::bfloat16_t>) ||
                     (FP32MatmulPrecision != 0 &&
                      std::same_as<PackedType, vecops::float32_t>)),
                "FP32 packing dtype does not match fp32_matmul_precision");
  static_assert(std::same_as<InputType, vecops::float32_t> ||
                    std::same_as<InputType, PackedType>,
                "non-FP32 weights must retain their dtype when packed");
  using MatmulAtom = typename SelectMatmulAtom<PackedType>::type;

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
  auto phase = workspace.serial_scope("pack_linear_b");
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
