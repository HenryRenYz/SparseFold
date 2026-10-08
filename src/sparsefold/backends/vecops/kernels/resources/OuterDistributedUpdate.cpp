// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file OuterDistributedUpdate.cpp
 *  @brief Post-gather contraction for distributed outer-product mean.
 */

#include <algorithm>
#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/ops/Transpose.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using LocalExtent = spec::L;
constexpr auto M = spec::M;
constexpr auto N = spec::N;
constexpr auto D = spec::OuterChannels;
constexpr auto Q = spec::Q;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed outer update requires AMX or SME"
#endif

namespace {
constexpr nint_t LeftRowsPerTile = 4;
#if defined(ARCH_X86_FAMILY)
constexpr nint_t RightRowsPerTile = 4;
#else
constexpr nint_t RightRowsPerTile = 8;
#endif
using LeftRows = meta::Dynamic<1, 1, LeftRowsPerTile>;
constexpr nint_t MaxRightRows = decltype(N)::value < RightRowsPerTile
                                    ? decltype(N)::value
                                    : RightRowsPerTile;
using RightRows = meta::Dynamic<1, 1, MaxRightRows>;

template <TensorLike Packed> auto output_b(Packed packed) {
  const auto layout = matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
      make_layout(make_shape(Q, D * D)));
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data(), layout);
}

template <TensorLike Bias> auto bias_transform(Bias bias) {
  const auto *data = bias.data();
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [data](auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const auto bias_value =
            vec::load_convert(tag, data + coordinate[1],
                              context.active_option(tag), vec::opt::zero);
        return vec::add(tag, value, bias_value);
      });
}
} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto left,
                TensorLike auto global_right, TensorLike auto mask,
                TensorLike auto output_projection, TensorLike auto output_bias,
                TensorLike auto output, int64_t row_offset) {
  assert_shape(left, {vecops::any, M, vecops::any, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(global_right, {N, vecops::any, M, D},
               {vecops::any, vecops::any, D, cint<1>});
  assert_shape(mask, {vecops::any, M, N}, {vecops::any, N, cint<1>});
  assert_shape(output, {vecops::any, vecops::any, N, Q},
               {vecops::any, vecops::any, Q, cint<1>});
  const nint_t batch_count = left.size(0);
  const LocalExtent local_extent{left.size(2)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t msa_rows = static_cast<nint_t>(M);
  const nint_t sequence = static_cast<nint_t>(N);
  if (batch_count == 0 || local_rows == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  const nint_t left_tiles = ceil_div(local_rows, LeftRowsPerTile);
  const nint_t right_tiles = ceil_div(sequence, RightRowsPerTile);
  const auto max_left_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
          make_layout(make_shape(cint<LeftRowsPerTile> * D, M)));
  const nint_t max_left_elements = numel(max_left_layout);
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto transpose_op = ops::transpose(ops::TransposeConfig<ComputeType>{});
  auto packer =
      ops::matmul_pack(ops::MatmulPackConfig<MatmulAtom, matmul::Operand::B>{});

  auto persistent = workspace.serial_scope("outer_distributed_persistent");
  const auto packed_left_shape =
      make_shape(batch_count, left_tiles, max_left_elements);
  const auto right_t_shape = make_shape(batch_count, N, D, M);
  const auto norm_shape = make_shape(batch_count, LocalExtent{local_rows}, N);
  auto packed_left =
      workspace.tensor<vecops::bfloat16_t>("packed_left", packed_left_shape);
  auto right_t = workspace.tensor<vecops::bfloat16_t>("right_t", right_t_shape);
  auto norm = workspace.tensor<vecops::bfloat16_t>("norm", norm_shape);

  const nint_t left_tasks = batch_count * left_tiles;
  const auto left_tmp_shape = make_shape(cint<LeftRowsPerTile> * D, M);
  auto left_tmp_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("left_t",
                                                               left_tmp_shape);
  const execution::TaskContext<Parallelism> sample_task{0};
  auto sample_left_storage = sample_task.local(left_tmp_storage);
  const LeftRows sample_left_rows{std::min(LeftRowsPerTile, local_rows)};
  auto sample_left_source = make_tensor(
      left.data(), make_layout(make_shape(M, sample_left_rows * D),
                               make_strides(left.stride(1), cint<1>)));
  auto sample_left_t =
      make_tensor(sample_left_storage.data(),
                  make_layout(make_shape(sample_left_rows * D, M)));
  const auto sample_packed_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
          make_layout(make_shape(sample_left_rows * D, M)));
  auto sample_packed_left =
      make_tensor(packed_left.data(), sample_packed_layout);
  auto prepared_left_transpose = transpose_op.template prepare<Parallelism>(
      workspace, "left_transpose_scratch", tensor::unbind(sample_left_source),
      tensor::unbind(sample_left_t));
  auto prepared_left_pack = packer.template prepare<Parallelism>(
      workspace, "left_pack_scratch", tensor::unbind(sample_left_t),
      tensor::unbind(sample_packed_left));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, left_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / left_tiles;
        const nint_t tile = task % left_tiles;
        const nint_t begin = tile * LeftRowsPerTile;
        const nint_t end = std::min(begin + LeftRowsPerTile, local_rows);
        const LeftRows rows{end - begin};
        auto left_tmp_storage_local = task_context.local(left_tmp_storage);
        auto source = make_tensor(
            left.data() + batch * left.stride(0) + begin * left.stride(2),
            make_layout(make_shape(M, rows * D),
                        make_strides(left.stride(1), cint<1>)));
        auto transposed = make_tensor(left_tmp_storage_local.data(),
                                      make_layout(make_shape(rows * D, M)));
        const auto active_layout =
            matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
                make_layout(make_shape(rows * D, M)));
        auto destination =
            make_tensor(packed_left.data() +
                            (batch * left_tiles + tile) * max_left_elements,
                        active_layout);
        prepared_left_transpose(task_context, source, transposed);
        prepared_left_pack(task_context, transposed, destination);
      });

  const nint_t right_tasks = batch_count * sequence;
  auto prepared_right_transpose = transpose_op.template prepare<Parallelism>(
      workspace, "right_transpose_scratch",
      tensor::unbind(global_right(0, 0, reserve, reserve)),
      tensor::unbind(right_t(0, 0, reserve, reserve)));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, right_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / sequence;
        const nint_t row = task % sequence;
        prepared_right_transpose(task_context,
                                 global_right(row, batch, reserve, reserve),
                                 right_t(batch, row, reserve, reserve));
      });
  workspace.parallel_for<Parallelism>(
      nint_t{0}, batch_count * local_rows, cint<1>,
      [&](execution::TaskContext<Parallelism>, const auto &work_item)
          VECOPS_INLINE_LAMBDA_NOEXCEPT {
            const nint_t task = work_item.begin();
            const nint_t batch = task / local_rows;
            const nint_t local = task % local_rows;
            const nint_t global_left = row_offset + local;
            for (nint_t right = 0; right < sequence; ++right) {
              nint_t count = 0;
              for (nint_t msa = 0; msa < msa_rows; ++msa)
                count += static_cast<nint_t>(mask(batch, msa, global_left) &&
                                             mask(batch, msa, right));
              norm(batch, local, right) =
                  vecops::bfloat16_t(static_cast<ComputeType>(count));
            }
          });

  const nint_t tasks = batch_count * left_tiles * right_tiles;
  const auto raw_shape =
      make_shape(cint<MaxRightRows> * D, cint<LeftRowsPerTile> * D);
  const auto reordered_shape =
      make_shape(cint<LeftRowsPerTile * MaxRightRows>, D * D);
  const auto projected_shape =
      make_shape(cint<LeftRowsPerTile * MaxRightRows>, Q);
  auto outer_phase = workspace.serial_scope("outer_distributed_tiles");
  auto raw_storage = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "raw", raw_shape);
  auto reordered_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("reordered",
                                                               reordered_shape);
  auto projected_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("projected",
                                                               projected_shape);
  const LeftRows sample_outer_left{std::min(LeftRowsPerTile, local_rows)};
  const RightRows sample_outer_right{std::min(RightRowsPerTile, sequence)};
  auto sample_raw_storage = sample_task.local(raw_storage);
  auto sample_reordered_storage = sample_task.local(reordered_storage);
  auto sample_projected_storage = sample_task.local(projected_storage);
  const auto sample_active_left_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
          make_layout(make_shape(sample_outer_left * D, M)));
  auto sample_active_left =
      make_tensor(packed_left.data(), sample_active_left_layout);
  auto sample_active_right = make_tensor(
      right_t.data(), make_layout(make_shape(sample_outer_right * D, M)));
  auto sample_raw = make_tensor(
      sample_raw_storage.data(),
      make_layout(make_shape(sample_outer_right * D, sample_outer_left * D),
                  make_strides(cint<LeftRowsPerTile> * D, cint<1>)));
  auto sample_reordered = make_tensor(
      sample_reordered_storage.data(),
      make_layout(make_shape(sample_outer_left * sample_outer_right, D * D)));
  auto sample_projected = make_tensor(
      sample_projected_storage.data(),
      make_layout(make_shape(sample_outer_left * sample_outer_right, Q)));
  auto prepared_outer_product = matmul_op.template prepare<Parallelism>(
      workspace, "outer_product_scratch", sample_outer_right * D,
      sample_outer_left * D, M, tensor::unbind(sample_active_right),
      tensor::unbind(sample_active_left), tensor::unbind(sample_raw));
  auto prepared_output_projection = matmul_op.template prepare<Parallelism>(
      workspace, "output_projection_scratch",
      sample_outer_left * sample_outer_right, Q, D * D,
      tensor::unbind(sample_reordered),
      tensor::unbind(output_b(output_projection)),
      tensor::unbind(tensor::output<ComputeType>(sample_projected,
                                                 bias_transform(output_bias))));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t per_batch = left_tiles * right_tiles;
        const nint_t batch = task / per_batch;
        const nint_t within = task % per_batch;
        const nint_t left_tile = within / right_tiles;
        const nint_t right_tile = within % right_tiles;
        const nint_t left_begin = left_tile * LeftRowsPerTile;
        const nint_t right_begin = right_tile * RightRowsPerTile;
        const LeftRows left_count{
            std::min(LeftRowsPerTile, local_rows - left_begin)};
        const RightRows right_count{
            std::min(RightRowsPerTile, sequence - right_begin)};
        const auto active_left_layout =
            matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
                make_layout(make_shape(left_count * D, M)));
        auto active_left =
            make_tensor(packed_left.data() + (batch * left_tiles + left_tile) *
                                                 max_left_elements,
                        active_left_layout);
        auto active_right =
            make_tensor(right_t.data() + batch * right_t.stride(0) +
                            right_begin * right_t.stride(1),
                        make_layout(make_shape(right_count * D, M)));
        auto raw_local = task_context.local(raw_storage);
        auto reordered_local = task_context.local(reordered_storage);
        auto projected_local = task_context.local(projected_storage);
        auto raw = make_tensor(
            raw_local.data(),
            make_layout(make_shape(right_count * D, left_count * D),
                        make_strides(cint<LeftRowsPerTile> * D, cint<1>)));
        auto reordered = make_tensor(
            reordered_local.data(),
            make_layout(make_shape(left_count * right_count, D * D)));
        auto projected =
            make_tensor(projected_local.data(),
                        make_layout(make_shape(left_count * right_count, Q)));
        prepared_outer_product(task_context, right_count * D, left_count * D, M,
                               active_right, active_left, raw);
        const nint_t lc = static_cast<nint_t>(left_count);
        const nint_t rc = static_cast<nint_t>(right_count);
        const nint_t dc = static_cast<nint_t>(D);
        for (nint_t li = 0; li < lc; ++li)
          for (nint_t rj = 0; rj < rc; ++rj)
            for (nint_t d2 = 0; d2 < dc; ++d2)
              std::copy_n(raw.data() + (rj * dc + d2) * raw.stride(0) + li * dc,
                          dc,
                          reordered.data() + ((li * rc + rj) * dc + d2) * dc);
        prepared_output_projection(task_context, left_count * right_count, Q,
                                   D * D, reordered,
                                   output_b(output_projection),
                                   tensor::output<ComputeType>(
                                       projected, bias_transform(output_bias)));
        using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
        using ValueTag = vec::Rebind<ComputeType, StorageTag>;
        const nint_t lanes = vec::size(ValueTag{});
        const auto epsilon = static_cast<ComputeType>(1e-3);
        for (nint_t li = 0; li < lc; ++li) {
          for (nint_t rj = 0; rj < rc; ++rj) {
            const auto inverse = vec::fill(
                ValueTag{},
                static_cast<ComputeType>(1) /
                    (epsilon + static_cast<ComputeType>(norm(
                                   batch, left_begin + li, right_begin + rj))));
            auto *destination = output.data() + batch * output.stride(0) +
                                (left_begin + li) * output.stride(1) +
                                (right_begin + rj) * output.stride(2);
            const auto *source =
                projected.data() + (li * rc + rj) * static_cast<nint_t>(Q);
            for (nint_t column = 0; column < static_cast<nint_t>(Q);
                 column += lanes) {
              const nint_t active =
                  std::min(lanes, static_cast<nint_t>(Q) - column);
              auto value =
                  vec::load_convert(ValueTag{}, source + column,
                                    vec::opt::first(active), vec::opt::zero);
              vec::store_convert(ValueTag{}, destination + column,
                                 vec::mul(ValueTag{}, value, inverse),
                                 vec::opt::first(active));
            }
          }
        }
      });
  outer_phase.close();
  persistent.close();
}
