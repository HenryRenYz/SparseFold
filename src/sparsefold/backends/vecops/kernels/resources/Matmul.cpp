// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file Matmul.cpp @brief Raw dynamic matmul routing with fused C input. */

#include <algorithm>
#include <concepts>
#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
using WeightBatchExtent = spec::WeightBatch;
using MExtent = spec::M;
using NExtent = spec::N;
using KExtent = spec::K;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
template <typename InputType> struct SelectMatmulAtom {
  static_assert(std::same_as<InputType, vecops::bfloat16_t> ||
                    std::same_as<InputType, vecops::float16_t>,
                "AMX raw matmul only supports BF16 and FP16");
  using type = std::conditional_t<std::same_as<InputType, vecops::bfloat16_t>,
                                  matmul::AMX_BF16F32, matmul::AMX_F16F32>;
};
#elif defined(HAS_SME)
template <typename InputType> struct SelectMatmulAtom {
  static_assert(std::same_as<InputType, vecops::bfloat16_t> ||
                    std::same_as<InputType, vecops::float16_t> ||
                    std::same_as<InputType, vecops::float32_t>,
                "SME raw matmul only supports BF16, FP16 and FP32");
  using type = std::conditional_t<
      std::same_as<InputType, vecops::bfloat16_t>, matmul::SME_BF16F32,
      std::conditional_t<std::same_as<InputType, vecops::float16_t>,
                         matmul::SME_F16F32, matmul::SME_F32F32>>;
};
#else
#error "Raw Matmul requires an AMX or SME matmul target"
#endif

namespace {
constexpr nint_t MinRowsPerTask = 128;

auto scale_output_transform(float alpha) {
  return make_elementwise_vec_transform<ComputeType, ComputeType>(
      [alpha]<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
          VECOPS_KERNEL_LAMBDA {
            return vec::mul(tag, value, vec::fill(tag, alpha));
          });
}

auto fused_c_output_transform(TensorLike auto c_input, float alpha,
                              float beta) {
  const auto *c_data = c_input.data();
  const nint_t row_stride = c_input.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [c_data, row_stride, alpha,
       beta](auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        const nint_t row = context.lane_coord(0)[0];
        const nint_t column = context.lane_coord(0)[1];
        const auto c_value =
            vec::load_convert(tag, c_data + row * row_stride + column,
                              context.active_option(tag), vec::opt::zero);
        return vec::add(tag, vec::mul(tag, value, vec::fill(tag, alpha)),
                        vec::mul(tag, c_value, vec::fill(tag, beta)));
      });
}
} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto input,
                TensorLike auto weight, auto c_input, TensorLike auto output,
                float alpha, float beta) {
  const BatchExtent batch{input.size(0)};
  const WeightBatchExtent weight_batch{weight.size(0)};
  const MExtent m{input.size(1)};
  const NExtent n{weight.size(1)};
  const KExtent k{input.size(2)};
  assert_shape(input, {batch, m, k}, {vecops::any, k, cint<1>});
  assert_shape(weight, {weight_batch, n, k}, {vecops::any, k, cint<1>});
  assert_shape(output, {batch, m, n}, {vecops::any, n, cint<1>});
  if constexpr (!tensor::is_nullopt_v<decltype(c_input)>)
    assert_shape(c_input, {batch, m, n},
                 {vecops::any, vecops::any, vecops::any});
  VECOPS_ASSERT(static_cast<nint_t>(weight_batch) == 1 ||
                    static_cast<nint_t>(weight_batch) ==
                        static_cast<nint_t>(batch),
                "weight batch must be one or match input batch");
  if (static_cast<nint_t>(batch) == 0 || static_cast<nint_t>(m) == 0)
    return;

  using InputType = std::remove_const_t<
      typename std::remove_cvref_t<decltype(input)>::ElementType>;
  using WeightType = std::remove_const_t<
      typename std::remove_cvref_t<decltype(weight)>::ElementType>;
  static_assert(std::same_as<InputType, WeightType>,
                "raw matmul operands must have the same dtype");
  using MatmulAtom = typename SelectMatmulAtom<InputType>::type;
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  // Every row task invokes matmul independently, and therefore packs its B
  // operand independently.  Do not create more row tasks than are needed to
  // fill the worker team: batched attention already exposes hundreds of
  // independent B operands and splitting M would only repack each one.
  const nint_t batch_count = static_cast<nint_t>(batch);
  const nint_t max_row_tasks = ceil_div(workspace.parallelism(), batch_count);
  const nint_t row_tasks =
      std::min(ceil_div(static_cast<nint_t>(m), MinRowsPerTask), max_row_tasks);
  const nint_t rows_per_task = ceil_div(static_cast<nint_t>(m), row_tasks);
  const nint_t task_count = static_cast<nint_t>(batch) * row_tasks;

  // Build one address-free operation pattern at the maximum row-task extent.
  // Dynamic tail views rebind to this capacity while preserving their active
  // row count at execution. Stateful epilogue data is deliberately stripped
  // from the pattern and supplied by each bound OutputSpec below.
  auto pattern_a = tensor::unbind(input(0, range(0, rows_per_task), reserve));
  auto pattern_b = tensor::unbind(weight(0, reserve, reserve));
  auto pattern_output_tensor = output(0, range(0, rows_per_task), reserve);
  auto pattern_output = [&] {
    if constexpr (!tensor::is_nullopt_v<decltype(c_input)>) {
      auto pattern_c = c_input(0, range(0, rows_per_task), reserve);
      return tensor::unbind(tensor::output<ComputeType>(
          pattern_output_tensor,
          fused_c_output_transform(pattern_c, alpha, beta)));
    } else {
      return tensor::unbind(tensor::output<ComputeType>(
          pattern_output_tensor, scale_output_transform(alpha)));
    }
  }();

  auto matmul_scope = workspace.serial_scope("raw_matmul");
  auto prepared_matmul = matmul_op.template prepare<Parallelism>(
      workspace, "matmul", rows_per_task, n, k, std::move(pattern_a),
      std::move(pattern_b), std::move(pattern_output));
  workspace.parallel_lanes<Parallelism>(
      [&](execution::TaskContext<Parallelism> task_context) {
        for (nint_t task = task_context.lane_id(); task < task_count;
             task += task_context.lane_count()) {
          const nint_t batch_index = task / row_tasks;
          const nint_t row_task = task % row_tasks;
          const nint_t begin = row_task * rows_per_task;
          const nint_t end =
              std::min(begin + rows_per_task, static_cast<nint_t>(m));
          const nint_t rows = end - begin;
          const nint_t weight_index =
              static_cast<nint_t>(weight_batch) == 1 ? 0 : batch_index;
          auto active_a = input(batch_index, range(begin, end), reserve);
          auto active_b = weight(weight_index, reserve, reserve);
          auto active_output_tensor =
              output(batch_index, range(begin, end), reserve);
          if constexpr (!tensor::is_nullopt_v<decltype(c_input)>) {
            auto active_c = c_input(batch_index, range(begin, end), reserve);
            auto active_output = tensor::output<ComputeType>(
                active_output_tensor,
                fused_c_output_transform(active_c, alpha, beta));
            prepared_matmul(task_context, rows, n, k, active_a, active_b,
                            active_output);
          } else {
            auto active_output = tensor::output<ComputeType>(
                active_output_tensor, scale_output_transform(alpha));
            prepared_matmul(task_context, rows, n, k, active_a, active_b,
                            active_output);
          }
        }
      });
  matmul_scope.close();
}
