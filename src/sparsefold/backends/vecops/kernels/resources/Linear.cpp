// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file Linear.cpp @brief Packed Linear with fused bias/residual/post-op. */

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/OptionalOperand.h"

using namespace vecops;
using namespace vecops::tensor;

using MExtent = spec::M;
using NExtent = spec::N;
using KExtent = spec::K;
using PExtent = spec::P;
constexpr auto FP32MatmulPrecision = spec::FP32MatmulPrecision;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
template <typename PackedType> struct SelectMatmulAtom {
  static_assert(std::same_as<PackedType, vecops::bfloat16_t> ||
                    std::same_as<PackedType, vecops::float16_t>,
                "AMX has no FP32 matmul atom; FP32 Linear requires "
                "fp32_matmul_precision=medium");
  using type = std::conditional_t<std::same_as<PackedType, vecops::bfloat16_t>,
                                  matmul::AMX_BF16F32, matmul::AMX_F16F32>;
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
#error "Linear requires an AMX or SME matmul target"
#endif

namespace {

constexpr nint_t RowsPerTask = 128;

VECOPS_INLINE auto apply_post_op(auto tag, auto value, int32_t post_op) {
  if (post_op == 1) {
    return vec::max(tag, value, vec::zeros(tag));
  } else if (post_op == 2) {
    const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(tag, value);
    return vec::mul(tag, value, sigmoid);
  } else if (post_op == 3) {
    return sparsefold_vecops::stable_sigmoid<ComputeType>(tag, value);
  }
  return value;
}

template <TensorLike Bias>
auto bias_post_transform(Bias bias, int32_t post_op) {
  const auto *bias_data = bias.data();
  return with_transform_store_mode<TransformStoreMode::coalesced>(
      make_lane_local_vec_transform<ComputeType, ComputeType>(
          [bias_data, post_op](auto tag, auto value,
                               const auto &context) VECOPS_KERNEL_LAMBDA {
            VECOPS_ASSERT(context.vector_axis == 1,
                          "Linear epilogue must traverse columns");
            const nint_t column = context.lane_coord(0)[1];
            const auto bias_value =
                vec::load_convert(tag, bias_data + column,
                                  context.active_option(tag), vec::opt::zero);
            return apply_post_op(tag, vec::add(tag, value, bias_value),
                                 post_op);
          }));
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto input,
                TensorLike auto packed_weight, TensorLike auto bias,
                auto residual, TensorLike auto output, int32_t post_op) {
  const MExtent m{input.size(0)};
  const NExtent n{output.size(1)};
  const KExtent k{input.size(1)};
  const PExtent p{packed_weight.size(0)};
  assert_shape(input, {m, k}, {vecops::any, cint<1>});
  assert_shape(packed_weight, {p}, {cint<1>});
  assert_shape(output, {m, n}, {n, cint<1>});
  assert_shape(bias, {n}, {vecops::any});
  if constexpr (!tensor::is_nullopt_v<decltype(residual)>)
    assert_shape(residual, {m, n}, {vecops::any, cint<1>});

  using PackedType = std::remove_const_t<
      typename std::remove_cvref_t<decltype(packed_weight)>::ElementType>;
  using MatmulAtom = typename SelectMatmulAtom<PackedType>::type;
  static_assert(!std::same_as<PackedType, vecops::float32_t> ||
                    FP32MatmulPrecision != 0,
                "FP32 packed weights require high/highest precision");
  static_assert(
      std::same_as<PackedType, vecops::float32_t> || FP32MatmulPrecision == 0 ||
          !std::same_as<std::remove_const_t<typename std::remove_cvref_t<
                            decltype(input)>::ElementType>,
                        vecops::float32_t>,
      "FP32 high/highest Linear requires FP32 packed weights");

  const nint_t row_count = input.size(0);
  if (row_count == 0)
    return;
  const auto logical_layout = make_layout(make_shape(n, k));
  const auto packed_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed_weight.size(0) >= numel(packed_layout),
                "packed Linear weight is too small");
  auto weight = make_tensor(packed_weight.data(), packed_layout);
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const nint_t task_count = ceil_div(row_count, RowsPerTask);
  const nint_t max_threads = workspace.parallelism();
  // Divide rows in native matmul blocks even when the established 128-row
  // task count is used to choose the team size. A strided assignment of whole
  // 128-row tasks leaves the SME team badly imbalanced as soon as task_count
  // is not divisible by the (typically much larger) ARM worker count -- for
  // example M=6144 gives ten workers 256 rows and the rest 128 rows. Balanced
  // native blocks retain packed-layout alignment and require one matmul entry
  // per worker.
  const nint_t SchedulingRows = MatmulAtom::M_R;
  const nint_t row_blocks = ceil_div(row_count, SchedulingRows);
#if defined(ARCH_X86_FAMILY)
  // Once enough rows are available, expose every AMX row tile to the worker
  // pool instead of limiting parallelism to one worker per 128 rows. This is
  // particularly important for M=640: five 128-row tasks underutilize an
  // eight-thread caller, while eight balanced 80-row ranges retain complete
  // 16-row AMX tiles. Each worker enters matmul only once, avoiding repeated
  // dynamic planning and TILECFG traversal setup on large-M projections. A
  // minimum of five AMX row tiles per worker prevents a 56-thread caller from
  // turning that same M=640 problem into forty tiny, overhead-bound regions.
  const nint_t RowsPerWorker = 5 * SchedulingRows;
  const nint_t column_count = static_cast<nint_t>(n);
  const nint_t reduction_size = static_cast<nint_t>(k);
  const bool compute_dense =
      reduction_size > 0 &&
      column_count >= ceil_div(nint_t{32768}, reduction_size);
  nint_t requested_threads;
  if (row_count >= 4 * RowsPerTask) {
    requested_threads =
        std::min(ceil_div(row_count, RowsPerWorker), max_threads);
  } else if (row_count >= RowsPerTask && compute_dense) {
    requested_threads = std::min(row_blocks, max_threads);
  } else {
    requested_threads = std::min(task_count, max_threads);
  }
  if (static_cast<nint_t>(n) < SchedulingRows ||
      static_cast<nint_t>(k) < SchedulingRows)
    requested_threads = std::min<nint_t>(requested_threads, 8);
#else
  // Preserve the established SME work granularity. The ARM path has a much
  // larger machine and thread count and is benchmarked with 128-row tasks.
  const nint_t requested_threads = std::min(task_count, max_threads);
#endif
  const nint_t sample_rows = std::min(
      row_count, ceil_div(row_blocks, requested_threads) * SchedulingRows);
  auto linear_scope = workspace.serial_scope("linear");
  auto sample_input = input(range(0, sample_rows), reserve);
  auto sample_output_tensor = output(range(0, sample_rows), reserve);
  auto prepared_matmul = [&] {
    auto output_pattern = tensor::unbind(tensor::output<ComputeType>(
        sample_output_tensor, bias_post_transform(bias, post_op)));
    if constexpr (!tensor::is_nullopt_v<decltype(residual)>) {
      return matmul_op.template prepare<Parallelism>(
          workspace, "matmul_scratch", sample_rows, n, k,
          tensor::unbind(sample_input), tensor::unbind(weight),
          tensor::unbind(tensor::input<ComputeType>(
              residual(range(0, sample_rows), reserve))),
          std::move(output_pattern));
    } else {
      return matmul_op.template prepare<Parallelism>(
          workspace, "matmul_scratch", sample_rows, n, k,
          tensor::unbind(sample_input), tensor::unbind(weight),
          std::move(output_pattern));
    }
  }();

  workspace.parallel_for<Parallelism>(
      nint_t{0}, requested_threads, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t worker = work_item.begin();
        const nint_t block_begin = row_blocks * worker / requested_threads;
        const nint_t block_end =
            row_blocks * (worker + 1) / requested_threads;
        if (block_begin == block_end)
          return;
        const nint_t begin = block_begin * SchedulingRows;
        const nint_t end = std::min(block_end * SchedulingRows, row_count);
        const nint_t rows = end - begin;
        auto active_input = input(range(begin, end), reserve);
        auto active_output_tensor = output(range(begin, end), reserve);
        auto transformed_output = tensor::output<ComputeType>(
            active_output_tensor, bias_post_transform(bias, post_op));
        if constexpr (!tensor::is_nullopt_v<decltype(residual)>) {
          prepared_matmul(
              task_context, rows, n, k, active_input, weight,
              tensor::input<ComputeType>(residual(range(begin, end), reserve)),
              transformed_output);
        } else {
          prepared_matmul(task_context, rows, n, k, active_input, weight,
                          transformed_output);
        }
      });
  linear_scope.close();
}
