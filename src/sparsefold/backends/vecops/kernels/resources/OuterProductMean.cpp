// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file OuterProductMean.cpp @brief AlphaFold 3 outer-product mean. */

#include <algorithm>
#include <bit>
#include <cstdint>
#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/Parallel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/ops/Transpose.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
constexpr auto M = spec::M;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto D = spec::OuterChannels;
constexpr auto Q = spec::Q;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "OuterProductMean requires an AMX or SME BF16 matmul target"
#endif

namespace {

constexpr nint_t ProjectionRows = 128;
#if defined(ARCH_X86_FAMILY)
constexpr nint_t LeftRowsPerTile = 4;
constexpr nint_t RightRowsPerTile = 4;
#else
constexpr nint_t LeftRowsPerTile = 4;
constexpr nint_t RightRowsPerTile = 8;
#endif

constexpr nint_t MaxProjectionRows = decltype(M * N)::value < ProjectionRows
                                         ? decltype(M * N)::value
                                         : ProjectionRows;
using ActiveProjectionRows = meta::Dynamic<1, 1, MaxProjectionRows>;
using ProjectionTileRows =
    std::conditional_t<(decltype(M * N)::value <= ProjectionRows) ||
                           (decltype(M * N)::value % ProjectionRows == 0),
                       meta::Const<MaxProjectionRows>, ActiveProjectionRows>;

constexpr nint_t MaxLeftRows =
    decltype(N)::value < LeftRowsPerTile ? decltype(N)::value : LeftRowsPerTile;
constexpr nint_t MaxRightRows = decltype(N)::value < RightRowsPerTile
                                    ? decltype(N)::value
                                    : RightRowsPerTile;
using ActiveLeftRows = meta::Dynamic<1, 1, MaxLeftRows>;
using ActiveRightRows = meta::Dynamic<1, 1, MaxRightRows>;
using LeftTileRows =
    std::conditional_t<(decltype(N)::value <= LeftRowsPerTile) ||
                           (decltype(N)::value % LeftRowsPerTile == 0),
                       meta::Const<MaxLeftRows>, ActiveLeftRows>;
using RightTileRows =
    std::conditional_t<(decltype(N)::value <= RightRowsPerTile) ||
                           (decltype(N)::value % RightRowsPerTile == 0),
                       meta::Const<MaxRightRows>, ActiveRightRows>;

template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, nint_t index, Spatial spatial,
              Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout),
                "packed B operand is too small");
  return make_tensor(packed.data() + index * packed.stride(0), layout);
}

template <TensorLike Mask>
auto mask_projection_transform(Mask mask, nint_t row_begin) {
  const auto *mask_data = mask.data();
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [mask_data, row_begin](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            VECOPS_ASSERT(context.vector_axis == 1,
                          "outer projection epilogue must traverse channels");
            const auto coordinate = context.lane_coord(0);
            const auto scale =
                vec::fill(tag, static_cast<ComputeType>(static_cast<unsigned>(
                                   mask_data[row_begin + coordinate[0]])));
            return vec::mul(tag, value, scale);
          });
}

template <TensorLike Bias> auto bias_transform(Bias bias) {
  const auto *bias_data = bias.data();
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [bias_data](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            VECOPS_ASSERT(context.vector_axis == 1,
                          "outer output epilogue must traverse channels");
            const auto coordinate = context.lane_coord(0);
            const auto bias_value =
                vec::load_convert(tag, bias_data + coordinate[1],
                                  context.active_option(tag), vec::opt::zero);
            return vec::add(tag, value, bias_value);
          });
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto mask, TensorLike auto norm_scale,
                TensorLike auto norm_offset, TensorLike auto left_projection,
                TensorLike auto right_projection,
                TensorLike auto output_projection, TensorLike auto output_bias,
                TensorLike auto output) {
  static_assert(decltype(M)::value > 0,
                "outer-product mean requires at least one MSA row");
  assert_shape(act, {vecops::any, M, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(mask, {vecops::any, M, N}, {vecops::any, N, cint<1>});
  assert_shape(norm_scale, {C}, {cint<1>});
  assert_shape(norm_offset, {C}, {cint<1>});
  assert_shape(left_projection, {cint<1>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(right_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output_bias, {Q}, {cint<1>});
  assert_shape(output, {vecops::any, N, N, Q},
               {vecops::any, vecops::any, Q, cint<1>});

  const auto batch_extent = BatchExtent{act.size(0)};
  const nint_t batch_count = act.size(0);
  const nint_t msa_rows = static_cast<nint_t>(M);
  const nint_t sequence = static_cast<nint_t>(N);
  if (batch_count == 0 || msa_rows == 0 || sequence == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const auto left_tiles = ceil_div(N, cint<LeftRowsPerTile>);
  const auto right_tiles = ceil_div(N, cint<RightRowsPerTile>);
  const nint_t left_tile_count = static_cast<nint_t>(left_tiles);
  const nint_t right_tile_count = static_cast<nint_t>(right_tiles);
  const auto max_left_matrix_shape = make_shape(cint<MaxLeftRows> * D, M);
  const auto max_left_packed_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
          make_layout(max_left_matrix_shape));
  const auto max_left_packed_elements = numel(max_left_packed_layout);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto transpose_op = ops::transpose(ops::TransposeConfig<ComputeType>{});
  auto packer =
      ops::matmul_pack(ops::MatmulPackConfig<MatmulAtom, matmul::Operand::B>{});

  auto left_projection_b = packed_b(left_projection, 0, D, C);
  auto right_projection_b = packed_b(right_projection, 0, D, C);
  auto output_projection_b = packed_b(output_projection, 0, Q, D * D);

  auto persistent_scope = workspace.serial_scope("persistent");
  const auto right_transposed_shape = make_shape(batch_extent, N, D, M);
  const auto packed_left_shape =
      make_shape(batch_extent, left_tiles, max_left_packed_elements);
  const auto norm_shape = make_shape(batch_extent, N, N);
  auto right_transposed = workspace.tensor<vecops::bfloat16_t>(
      "right_transposed", right_transposed_shape);
  auto packed_left =
      workspace.tensor<vecops::bfloat16_t>("packed_left", packed_left_shape);
  auto norm = workspace.tensor<vecops::bfloat16_t>("norm", norm_shape);
  const execution::TaskContext<Parallelism> sample_task{0};

  {
    auto projection_scope = workspace.serial_scope("projection");
    const auto projected_shape = make_shape(batch_extent, M, N, D);
    auto left_projected =
        workspace.tensor<vecops::bfloat16_t>("left_projected", projected_shape);
    auto right_projected = workspace.tensor<vecops::bfloat16_t>(
        "right_projected", projected_shape);

    const auto flattened_rows = M * N;
    const nint_t flat_row_count = static_cast<nint_t>(flattened_rows);
    const auto projection_tiles =
        ceil_div(flattened_rows, cint<ProjectionRows>);
    const nint_t projection_tile_count = static_cast<nint_t>(projection_tiles);
    const nint_t projection_tasks = batch_count * projection_tile_count;
    const auto normalized_tile_shape = make_shape(cint<MaxProjectionRows>, C);
    auto normalized_workers =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "normalized", normalized_tile_shape);

    auto act_rows = make_tensor(
        act.data(), make_layout(make_shape(batch_extent, flattened_rows, C)));
    auto mask_rows = make_tensor(
        mask.data(), make_layout(make_shape(batch_extent, flattened_rows)));
    auto left_rows =
        make_tensor(left_projected.data(),
                    make_layout(make_shape(batch_extent, flattened_rows, D)));
    auto right_rows =
        make_tensor(right_projected.data(),
                    make_layout(make_shape(batch_extent, flattened_rows, D)));

    auto sample_normalized_storage = sample_task.local(normalized_workers);
    const ProjectionTileRows planning_rows{
        std::min(flat_row_count, ProjectionRows)};
    auto planning_act = act_rows(0, range(0, planning_rows), reserve);
    auto planning_norm =
        sample_normalized_storage(range(0, planning_rows), reserve);
    auto planning_left = left_rows(0, range(0, planning_rows), reserve);
    auto planning_left_out = tensor::output<ComputeType>(
        planning_left, mask_projection_transform(mask_rows(0, reserve), 0));
    auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
        workspace, "layer_norm_scratch",
        tensor::unbind(tensor::input<ComputeType>(planning_act)),
        tensor::unbind(tensor::input<ComputeType>(norm_scale)),
        tensor::unbind(tensor::input<ComputeType>(norm_offset)),
        tensor::unbind(tensor::output<ComputeType>(planning_norm)));
    auto prepared_left_projection = matmul_op.template prepare<Parallelism>(
        workspace, "left_projection_scratch", planning_rows, D, C,
        tensor::unbind(planning_norm), tensor::unbind(left_projection_b),
        tensor::unbind(planning_left_out));
    auto prepared_right_projection = matmul_op.template prepare<Parallelism>(
        workspace, "right_projection_scratch", planning_rows, D, C,
        tensor::unbind(planning_norm), tensor::unbind(right_projection_b),
        tensor::unbind(planning_left_out));

    workspace.parallel_for<Parallelism>(
        nint_t{0}, projection_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          const nint_t batch = task / projection_tile_count;
          const nint_t tile = task % projection_tile_count;
          const nint_t begin = tile * ProjectionRows;
          const nint_t end = std::min(begin + ProjectionRows, flat_row_count);
          const ProjectionTileRows rows{end - begin};
          auto normalized = task_context.local(normalized_workers);
          auto active_act = act_rows(batch, range(begin, end), reserve);
          auto active_norm = normalized(range(0, rows), reserve);
          auto batch_mask = mask_rows(batch, reserve);
          auto active_left = left_rows(batch, range(begin, end), reserve);
          auto active_right = right_rows(batch, range(begin, end), reserve);
          auto left_output = tensor::output<ComputeType>(
              active_left, mask_projection_transform(batch_mask, begin));
          auto right_output = tensor::output<ComputeType>(
              active_right, mask_projection_transform(batch_mask, begin));

          prepared_layer_norm(task_context, active_act, norm_scale, norm_offset,
                              active_norm);
          prepared_left_projection(task_context, rows, D, C, active_norm,
                                   left_projection_b, left_output);
          prepared_right_projection(task_context, rows, D, C, active_norm,
                                    right_projection_b, right_output);
        });

    const nint_t left_tasks = batch_count * left_tile_count;
    const auto left_transpose_shape = make_shape(cint<MaxLeftRows> * D, M);
    auto left_t_workers =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "left_transpose", left_transpose_shape);
    const LeftTileRows planning_left_rows{std::min(sequence, LeftRowsPerTile)};
    auto planning_left_source =
        make_tensor(left_projected.data(),
                    make_layout(make_shape(M, planning_left_rows * D),
                                make_strides(N * D, cint<1>)));
    auto planning_left_storage = sample_task.local(left_t_workers);
    auto planning_left_t =
        make_tensor(planning_left_storage.data(),
                    make_layout(make_shape(planning_left_rows * D, M)));
    const auto planning_packed_left_layout =
        matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
            make_layout(make_shape(planning_left_rows * D, M)));
    auto prepared_left_transpose = transpose_op.template prepare<Parallelism>(
        workspace, "left_transpose_scratch",
        tensor::unbind(planning_left_source), tensor::unbind(planning_left_t));
    auto prepared_left_pack = packer.template prepare<Parallelism>(
        workspace, "left_pack_scratch", tensor::unbind(planning_left_t),
        tensor::unbind(
            make_tensor(packed_left.data(), planning_packed_left_layout)));

    workspace.parallel_for<Parallelism>(
        nint_t{0}, left_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          const nint_t batch = task / left_tile_count;
          const nint_t tile = task % left_tile_count;
          const nint_t begin = tile * LeftRowsPerTile;
          const nint_t end = std::min(begin + LeftRowsPerTile, sequence);
          const LeftTileRows rows{end - begin};
          auto left_t = task_context.local(left_t_workers);
          auto source = make_tensor(left_projected.data() +
                                        batch * left_projected.stride(0) +
                                        begin * static_cast<nint_t>(D),
                                    make_layout(make_shape(M, rows * D),
                                                make_strides(N * D, cint<1>)));
          auto active_left_t =
              make_tensor(left_t.data(), make_layout(make_shape(rows * D, M)));
          const auto active_packed_layout =
              matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
                  make_layout(make_shape(rows * D, M)));
          auto destination = make_tensor(packed_left.data() +
                                             (batch * left_tile_count + tile) *
                                                 max_left_packed_elements,
                                         active_packed_layout);
          prepared_left_transpose(task_context, source, active_left_t);
          prepared_left_pack(task_context, active_left_t, destination);
        });

    const nint_t right_tasks = batch_count * right_tile_count;
    const RightTileRows planning_right_rows{
        std::min(sequence, RightRowsPerTile)};
    auto planning_right_source =
        make_tensor(right_projected.data(),
                    make_layout(make_shape(M, planning_right_rows * D),
                                make_strides(N * D, cint<1>)));
    auto planning_right_destination =
        make_tensor(right_transposed.data(),
                    make_layout(make_shape(planning_right_rows * D, M)));
    auto prepared_right_transpose = transpose_op.template prepare<Parallelism>(
        workspace, "right_transpose_scratch",
        tensor::unbind(planning_right_source),
        tensor::unbind(planning_right_destination));
    workspace.parallel_for<Parallelism>(
        nint_t{0}, right_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          const nint_t batch = task / right_tile_count;
          const nint_t tile = task % right_tile_count;
          const nint_t begin = tile * RightRowsPerTile;
          const nint_t end = std::min(begin + RightRowsPerTile, sequence);
          const RightTileRows rows{end - begin};
          auto source = make_tensor(right_projected.data() +
                                        batch * right_projected.stride(0) +
                                        begin * static_cast<nint_t>(D),
                                    make_layout(make_shape(M, rows * D),
                                                make_strides(N * D, cint<1>)));
          auto destination = make_tensor(
              right_transposed.data() + batch * right_transposed.stride(0) +
                  begin * static_cast<nint_t>(D) * msa_rows,
              make_layout(make_shape(rows * D, M)));
          prepared_right_transpose(task_context, source, destination);
        });
    projection_scope.close();
  }

  {
    auto norm_scope = workspace.serial_scope("mask_norm");
    const auto mask_words = ceil_div(M, cint<64>);
    const auto mask_bits_shape = make_shape(batch_extent, N, mask_words);
    auto mask_bits = workspace.tensor<uint64_t>("mask_bits", mask_bits_shape);
    const nint_t word_count = static_cast<nint_t>(mask_words);

    workspace.parallel_for<Parallelism>(
        nint_t{0}, batch_count * sequence, cint<1>,
        [&](execution::TaskContext<Parallelism>, const auto &work_item)
            VECOPS_INLINE_LAMBDA_NOEXCEPT {
              const nint_t task = work_item.begin();
              const nint_t batch = task / sequence;
              const nint_t residue = task % sequence;
              for (nint_t word = 0; word < word_count; ++word)
                mask_bits(batch, residue, word) = 0;
              for (nint_t row = 0; row < msa_rows; ++row) {
                if (mask(batch, row, residue))
                  mask_bits(batch, residue, row / 64) |=
                      uint64_t{1} << static_cast<unsigned>(row % 64);
              }
            });

    workspace.parallel_for<Parallelism>(
        nint_t{0}, batch_count * sequence, cint<1>,
        [&](execution::TaskContext<Parallelism>, const auto &work_item)
            VECOPS_INLINE_LAMBDA_NOEXCEPT {
              const nint_t task = work_item.begin();
              const nint_t batch = task / sequence;
              const nint_t left = task % sequence;
              for (nint_t right = 0; right < sequence; ++right) {
                uint64_t count = 0;
                for (nint_t word = 0; word < word_count; ++word)
                  count += static_cast<uint64_t>(
                      std::popcount(mask_bits(batch, left, word) &
                                    mask_bits(batch, right, word)));
                norm(batch, left, right) =
                    vecops::bfloat16_t(static_cast<ComputeType>(count));
              }
            });
    norm_scope.close();
  }

  auto outer_scope = workspace.serial_scope("outer_tiles");
  const nint_t outer_tasks = batch_count * left_tile_count * right_tile_count;
  const auto raw_shape =
      make_shape(cint<MaxRightRows> * D, cint<MaxLeftRows> * D);
  const auto reordered_shape =
      make_shape(cint<MaxLeftRows * MaxRightRows>, D * D);
  const auto projected_shape = make_shape(cint<MaxLeftRows * MaxRightRows>, Q);
  auto raw_workers = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "raw", raw_shape);
  auto reordered_workers =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("reordered",
                                                               reordered_shape);
  auto projected_workers =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("projected",
                                                               projected_shape);

  const LeftTileRows planning_left_rows{std::min(sequence, LeftRowsPerTile)};
  const RightTileRows planning_right_rows{std::min(sequence, RightRowsPerTile)};
  auto planning_right =
      make_tensor(right_transposed.data(),
                  make_layout(make_shape(planning_right_rows * D, M)));
  const auto planning_left_packed_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
          make_layout(make_shape(planning_left_rows * D, M)));
  auto planning_left =
      make_tensor(packed_left.data(), planning_left_packed_layout);
  auto planning_raw_storage = sample_task.local(raw_workers);
  auto planning_reordered_storage = sample_task.local(reordered_workers);
  auto planning_projected_storage = sample_task.local(projected_workers);
  auto planning_raw = make_tensor(
      planning_raw_storage.data(),
      make_layout(make_shape(planning_right_rows * D, planning_left_rows * D),
                  make_strides(cint<MaxLeftRows> * D, cint<1>)));
  auto planning_reordered = make_tensor(
      planning_reordered_storage.data(),
      make_layout(make_shape(planning_left_rows * planning_right_rows, D * D)));
  auto planning_projected = make_tensor(
      planning_projected_storage.data(),
      make_layout(make_shape(planning_left_rows * planning_right_rows, Q)));
  auto planning_projected_output = tensor::output<ComputeType>(
      planning_projected, bias_transform(output_bias));
  auto prepared_outer_product = matmul_op.template prepare<Parallelism>(
      workspace, "outer_product_scratch", planning_right_rows * D,
      planning_left_rows * D, M, tensor::unbind(planning_right),
      tensor::unbind(planning_left), tensor::unbind(planning_raw));
  auto prepared_output_projection = matmul_op.template prepare<Parallelism>(
      workspace, "output_projection_scratch",
      planning_left_rows * planning_right_rows, Q, D * D,
      tensor::unbind(planning_reordered), tensor::unbind(output_projection_b),
      tensor::unbind(planning_projected_output));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, outer_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t tiles_per_batch = left_tile_count * right_tile_count;
        const nint_t batch = task / tiles_per_batch;
        const nint_t in_batch = task % tiles_per_batch;
        const nint_t left_tile = in_batch / right_tile_count;
        const nint_t right_tile = in_batch % right_tile_count;
        const nint_t left_begin = left_tile * LeftRowsPerTile;
        const nint_t right_begin = right_tile * RightRowsPerTile;
        const nint_t left_end =
            std::min(left_begin + LeftRowsPerTile, sequence);
        const nint_t right_end =
            std::min(right_begin + RightRowsPerTile, sequence);
        const LeftTileRows left_rows{left_end - left_begin};
        const RightTileRows right_rows{right_end - right_begin};
        const auto active_left_packed_layout =
            matmul::packed_layout<MatmulAtom, matmul::Operand::B>(
                make_layout(make_shape(left_rows * D, M)));
        auto active_left = make_tensor(
            packed_left.data() + (batch * left_tile_count + left_tile) *
                                     max_left_packed_elements,
            active_left_packed_layout);
        auto active_right = make_tensor(
            right_transposed.data() + batch * right_transposed.stride(0) +
                right_begin * static_cast<nint_t>(D) * msa_rows,
            make_layout(make_shape(right_rows * D, M)));
        auto raw_storage = task_context.local(raw_workers);
        auto reordered_storage = task_context.local(reordered_workers);
        auto projected_storage = task_context.local(projected_workers);
        auto raw = make_tensor(
            raw_storage.data(),
            make_layout(make_shape(right_rows * D, left_rows * D),
                        make_strides(cint<MaxLeftRows> * D, cint<1>)));
        auto reordered =
            make_tensor(reordered_storage.data(),
                        make_layout(make_shape(left_rows * right_rows, D * D)));
        auto projected =
            make_tensor(projected_storage.data(),
                        make_layout(make_shape(left_rows * right_rows, Q)));
        auto projected_output =
            tensor::output<ComputeType>(projected, bias_transform(output_bias));

        prepared_outer_product(task_context, right_rows * D, left_rows * D, M,
                               active_right, active_left, raw);

        const nint_t left_count = static_cast<nint_t>(left_rows);
        const nint_t right_count = static_cast<nint_t>(right_rows);
        const nint_t outer_channels = static_cast<nint_t>(D);
        for (nint_t left = 0; left < left_count; ++left) {
          for (nint_t right = 0; right < right_count; ++right) {
            for (nint_t d2 = 0; d2 < outer_channels; ++d2) {
              const auto *source =
                  raw.data() + (right * outer_channels + d2) * raw.stride(0) +
                  left * outer_channels;
              auto *destination =
                  reordered.data() +
                  ((left * right_count + right) * outer_channels + d2) *
                      outer_channels;
              std::copy_n(source, outer_channels, destination);
            }
          }
        }

        prepared_output_projection(task_context, left_rows * right_rows, Q,
                                   D * D, reordered, output_projection_b,
                                   projected_output);

        using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
        using ValueTag = vec::Rebind<ComputeType, StorageTag>;
        const nint_t lanes = vec::size(ValueTag{});
        const auto epsilon = static_cast<ComputeType>(1e-3);
        for (nint_t left = 0; left < left_count; ++left) {
          for (nint_t right = 0; right < right_count; ++right) {
            const auto denominator =
                epsilon + static_cast<ComputeType>(norm(
                              batch, left_begin + left, right_begin + right));
            const auto inverse = vec::fill(
                ValueTag{}, static_cast<ComputeType>(1) / denominator);
            const auto *source =
                projected.data() +
                (left * right_count + right) * static_cast<nint_t>(Q);
            auto *destination = output.data() + batch * output.stride(0) +
                                (left_begin + left) * output.stride(1) +
                                (right_begin + right) * output.stride(2);
            for (nint_t column = 0; column < static_cast<nint_t>(Q);
                 column += lanes) {
              const nint_t active =
                  std::min(lanes, static_cast<nint_t>(Q) - column);
              auto value =
                  vec::load_convert(ValueTag{}, source + column,
                                    vec::opt::first(active), vec::opt::zero);
              value = vec::mul(ValueTag{}, value, inverse);
              vec::store_convert(ValueTag{}, destination + column, value,
                                 vec::opt::first(active));
            }
          }
        }
      });
  outer_scope.close();
  persistent_scope.close();
}
