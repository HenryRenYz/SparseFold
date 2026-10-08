// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file TriangleMultiplication.cpp @brief Fused AlphaFold 3 triangle
 * multiplication. */

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "TriangleMultiplication requires an AMX or SME BF16 matmul target"
#endif

namespace {

constexpr nint_t RowsPerTask = 128;
constexpr nint_t MaxTileRows =
    decltype(N)::value < RowsPerTask ? decltype(N)::value : RowsPerTask;
using ActiveRows = meta::Dynamic<1, 1, MaxTileRows>;
using TileRows = std::conditional_t<(decltype(N)::value <= RowsPerTask) ||
                                        (decltype(N)::value % RowsPerTask == 0),
                                    meta::Const<MaxTileRows>, ActiveRows>;

template <typename Tag, typename Value>
VECOPS_INLINE auto tail_safe_sigmoid_vec(Tag tag, Value value) {
#if defined(HAS_SME)
  return sparsefold_vecops::polynomial_sigmoid<ComputeType>(tag, value);
#else
  return sparsefold_vecops::stable_sigmoid<ComputeType>(tag, value);
#endif
}

template <TensorLike Gate, TensorLike Mask>
auto gated_masked_transform(Gate gate, Mask mask, nint_t row_begin,
                            bool keep_row) {
  const auto *gate_data = gate.data();
  const auto *mask_data = mask.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, mask_data, gate_stride, row_begin, keep_row](
          auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        VECOPS_ASSERT(context.vector_axis == 1,
                      "triangle GLU epilogue must traverse channels");
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
#if defined(__x86_64__) || defined(_M_X64)
        const auto mask_scale = vec::fill(
            tag, static_cast<ComputeType>(
                     static_cast<unsigned>(keep_row) &
                     static_cast<unsigned>(mask_data[row_begin + row])));
#else
        // On the tested SVE target this branch is highly predictable and its
        // early-out is cheaper than adding a vector mask multiply to every
        // epilogue invocation.
        if (!keep_row || !mask_data[row_begin + row])
          return vec::zeros(tag);
#endif
        const auto gate_value = vec::load_convert(
            tag, gate_data + row * gate_stride + coordinate[1],
            context.active_option(tag), vec::opt::zero);
        const auto sigmoid = tail_safe_sigmoid_vec(tag, gate_value);
#if defined(__x86_64__) || defined(_M_X64)
        return vec::mul(tag, vec::mul(tag, value, sigmoid), mask_scale);
#else
        return vec::mul(tag, value, sigmoid);
#endif
      });
}

template <TensorLike Projected, TensorLike Residual>
auto gated_residual_transform(Projected projected, Residual residual) {
  const auto *projected_data = projected.data();
  const auto *residual_data = residual.data();
  const nint_t projected_stride = projected.stride(0);
  const nint_t residual_stride = residual.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [projected_data, residual_data, projected_stride, residual_stride](
          auto tag, auto gate, const auto &context) VECOPS_KERNEL_LAMBDA {
        VECOPS_ASSERT(context.vector_axis == 1,
                      "triangle output epilogue must traverse channels");
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto projected_value = vec::load_convert(
            tag, projected_data + row * projected_stride + column,
            context.active_option(tag), vec::opt::zero);
        const auto residual_value = vec::load_convert(
            tag, residual_data + row * residual_stride + column,
            context.active_option(tag), vec::opt::zero);
        const auto sigmoid = tail_safe_sigmoid_vec(tag, gate);
        return vec::add(tag, residual_value,
                        vec::mul(tag, projected_value, sigmoid));
      });
}

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

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto mask, TensorLike auto left_norm_scale,
                TensorLike auto left_norm_offset, TensorLike auto gw_proj,
                TensorLike auto gw_projection_b,
                TensorLike auto center_norm_scale,
                TensorLike auto center_norm_offset,
                TensorLike auto output_projection,
                TensorLike auto gating_linear, TensorLike auto output) {
  assert_shape(act, {vecops::any, N, N, C},
               {vecops::any, vecops::any, vecops::any, cint<1>});
  assert_shape(mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(left_norm_scale, {C}, {cint<1>});
  assert_shape(left_norm_offset, {C}, {cint<1>});
  assert_shape(gw_proj, {cint<2>, 2 * C, C}, {2 * C * C, C, cint<1>});
  assert_shape(gw_projection_b, {cint<4>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(center_norm_scale, {C}, {cint<1>});
  assert_shape(center_norm_offset, {C}, {cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(gating_linear, {cint<1>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, N, N, C},
               {vecops::any, vecops::any, vecops::any, cint<1>});

  const auto batch_extent = BatchExtent{act.size(0)};
  const nint_t batch_count = act.size(0);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t channels = static_cast<nint_t>(C);
  if (batch_count == 0 || sequence == 0)
    return;

  const auto activation_shape = make_shape(batch_extent, N, N, C);
  const auto projection_shape = make_shape(batch_extent, C, N, N);
  auto kernel_workspace = workspace.serial_scope("triangle_multiplication");
  auto normalized =
      workspace.tensor<vecops::bfloat16_t>("normalized", activation_shape);
  auto a = workspace.tensor<vecops::bfloat16_t>("a", projection_shape);
  auto b = workspace.tensor<vecops::bfloat16_t>("b", projection_shape);
  auto product =
      workspace.tensor<vecops::bfloat16_t>("product", projection_shape);

  auto gw_flat =
      make_tensor(gw_proj.data(),
                  make_layout(make_shape(4 * C, C), make_strides(C, cint<1>)));
  auto gate_a = packed_b(gw_projection_b, 0, C, C);
  auto value_a = packed_b(gw_projection_b, 1, C, C);
  auto gate_b = packed_b(gw_projection_b, 2, C, C);
  auto value_b = packed_b(gw_projection_b, 3, C, C);
  auto output_projection_b = packed_b(output_projection, 0, C, C);
  auto gating_linear_b = packed_b(gating_linear, 0, C, C);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});

  const nint_t row_tiles = ceil_div(sequence, RowsPerTask);
  const nint_t projection_tasks = batch_count * sequence * row_tiles;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
#if defined(__x86_64__) || defined(_M_X64)
  constexpr bool UseFusedProjection = true;
#else
  // SME loses more in the (4C,C) projection shape and the explicit vector
  // post-pass than it saves by merging four smaller matmuls.  Keep this a
  // target policy instead of forcing the x86 crossover onto every backend.
  constexpr bool UseFusedProjection = false;
#endif
  auto projection_workspace = workspace.serial_scope("projection");
  if constexpr (UseFusedProjection && decltype(N)::value >= 64) {
    // Phase 1: normalize and run all four interleaved gate/value projections
    // in one (4C,C) x (rows,C)^T matmul.  The temporary keeps rows contiguous
    // for each projected channel, matching the legacy fused projection while
    // the vector post-pass consumes and writes complete BF16 words.
    const TileRows tile_rows{std::min(sequence, RowsPerTask)};
    const auto projection_tile_shape = make_shape(4 * C, tile_rows);
    auto projected_all_workers =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "projected_all", projection_tile_shape);
    auto sample_act = act(0, 0, range(0, tile_rows), reserve);
    auto sample_norm = normalized(0, 0, range(0, tile_rows), reserve);
    auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
        workspace, "layer_norm_scratch",
        tensor::unbind(tensor::input<ComputeType>(sample_act)),
        tensor::unbind(tensor::input<ComputeType>(left_norm_scale)),
        tensor::unbind(tensor::input<ComputeType>(left_norm_offset)),
        tensor::unbind(tensor::output<ComputeType>(sample_norm)));
    auto prepared_projection = matmul_op.template prepare<Parallelism>(
        workspace, "projection_scratch", 4 * C, tile_rows, C,
        tensor::unbind(gw_flat), tensor::unbind(sample_norm),
        tensor::unbind(projected_all_workers(0, reserve, reserve)));

    workspace.parallel_for<Parallelism>(
        nint_t{0}, projection_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          auto projected_all = task_context.local(projected_all_workers);
          const nint_t tile = task % row_tiles;
          const nint_t matrix = task / row_tiles;
          const nint_t batch = matrix / sequence;
          const nint_t row_index = matrix % sequence;
          const nint_t begin = tile * RowsPerTask;
          const nint_t end = std::min(begin + RowsPerTask, sequence);
          const TileRows rows{end - begin};
          auto active_act = act(batch, row_index, range(begin, end), reserve);
          auto active_norm =
              normalized(batch, row_index, range(begin, end), reserve);
          auto active_projected_all = projected_all(reserve, range(0, rows));
          auto batch_mask = mask(batch, reserve);
          const bool keep_row = batch_mask(row_index);
          auto active_a = a(batch, reserve, row_index, range(begin, end));
          auto active_b = b(batch, reserve, row_index, range(begin, end));

          prepared_layer_norm(task_context, active_act, left_norm_scale,
                              left_norm_offset, active_norm);
          prepared_projection(task_context, 4 * C, rows, C, gw_flat,
                              active_norm, active_projected_all);

          using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
          using ValueTag = vec::Rebind<ComputeType, StorageTag>;
          using MaskTag = vec::Rebind<int8_t, ValueTag>;
          const nint_t lanes = vec::size(ValueTag{});
          const auto one = vec::fill(ValueTag{}, static_cast<ComputeType>(1));
          const auto zero = vec::zeros(ValueTag{});
          const auto row_scale =
              vec::fill(ValueTag{}, static_cast<ComputeType>(keep_row));
          const auto *mask_data =
              reinterpret_cast<const int8_t *>(batch_mask.data() + begin);
          for (nint_t channel = 0; channel < channels; ++channel) {
            const auto *gate_a_data =
                active_projected_all.data() +
                (2 * channel) * active_projected_all.stride(0);
            const auto *value_a_data =
                gate_a_data + active_projected_all.stride(0);
            const auto *gate_b_data =
                active_projected_all.data() +
                (2 * channels + 2 * channel) * active_projected_all.stride(0);
            const auto *value_b_data =
                gate_b_data + active_projected_all.stride(0);
            auto *a_data = active_a.data() + channel * active_a.stride(0);
            auto *b_data = active_b.data() + channel * active_b.stride(0);
            for (nint_t offset = 0; offset < rows; offset += lanes) {
              const nint_t active =
                  std::min(lanes, static_cast<nint_t>(rows) - offset);
              const auto apply_gate =
                  [&](const auto *gate_data,
                      const auto *value_data) VECOPS_INLINE_LAMBDA {
                    const auto gate_value = vec::load_convert(
                        ValueTag{}, gate_data + offset, vec::opt::first(active),
                        vec::opt::zero);
                    const auto value = vec::load_convert(
                        ValueTag{}, value_data + offset,
                        vec::opt::first(active), vec::opt::zero);
                    const auto sigmoid =
                        tail_safe_sigmoid_vec(ValueTag{}, gate_value);
                    auto result = vec::mul(ValueTag{}, value, sigmoid);
                    const auto mask_values =
                        vec::load(MaskTag{}, mask_data + offset,
                                  vec::opt::first(active), vec::opt::zero);
                    const auto mask_predicate =
                        vec::convert(ValueTag{}, MaskTag{},
                                     vec::cmpne(MaskTag{}, mask_values,
                                                vec::zeros(MaskTag{})));
                    result = vec::mul(
                        ValueTag{},
                        vec::blend(ValueTag{}, zero, mask_predicate, result),
                        row_scale);
                    return result;
                  };
              vec::store_convert(ValueTag{}, a_data + offset,
                                 apply_gate(gate_a_data, value_a_data),
                                 vec::opt::first(active));
              vec::store_convert(ValueTag{}, b_data + offset,
                                 apply_gate(gate_b_data, value_b_data),
                                 vec::opt::first(active));
            }
          }
        });
  } else {
    // Four narrow projections are cheaper on SME and for small N on x86.
    // Stage values contiguously before the gate/mask scatter: writing a matmul
    // transform directly through the transposed [C,N,N] view loses elements
    // on both backends for short, partial-vector rows.
    const TileRows tile_rows{std::min(sequence, RowsPerTask)};
    // Keep projection tiles channel-major.  The matmul writes through a
    // transposed view, while the sigmoid epilogue then walks contiguous rows
    // for each channel and stores directly into the transposed output view.
    const auto tile_shape = make_shape(C, tile_rows);
    auto gate_workers =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("gate",
                                                                 tile_shape);
    auto value_workers =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("value",
                                                                 tile_shape);
    auto sample_act = act(0, 0, range(0, tile_rows), reserve);
    auto sample_norm = normalized(0, 0, range(0, tile_rows), reserve);
    auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
        workspace, "layer_norm_scratch",
        tensor::unbind(tensor::input<ComputeType>(sample_act)),
        tensor::unbind(tensor::input<ComputeType>(left_norm_scale)),
        tensor::unbind(tensor::input<ComputeType>(left_norm_offset)),
        tensor::unbind(tensor::output<ComputeType>(sample_norm)));
    auto prepared_projection = matmul_op.template prepare<Parallelism>(
        workspace, "projection_scratch", tile_rows, C, C,
        tensor::unbind(sample_norm), tensor::unbind(gate_a),
        tensor::unbind(transpose<0, 1>(gate_workers(0, reserve, reserve))));

    workspace.parallel_for<Parallelism>(
        nint_t{0}, projection_tasks, cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          auto gate = task_context.local(gate_workers);
          auto value = task_context.local(value_workers);
          const nint_t tile = task % row_tiles;
          const nint_t matrix = task / row_tiles;
          const nint_t batch = matrix / sequence;
          const nint_t row_index = matrix % sequence;
          const nint_t begin = tile * RowsPerTask;
          const nint_t end = std::min(begin + RowsPerTask, sequence);
          const TileRows rows{end - begin};
          auto active_act = act(batch, row_index, range(begin, end), reserve);
          auto active_norm =
              normalized(batch, row_index, range(begin, end), reserve);
          auto active_gate =
              transpose<0, 1>(gate(reserve, range(0, rows)));
          auto active_value =
              transpose<0, 1>(value(reserve, range(0, rows)));
          auto batch_mask = mask(batch, reserve);
          const bool keep_row = batch_mask(row_index);
          auto active_a =
              transpose<0, 1>(a(batch, reserve, row_index, range(begin, end)));
          auto active_b =
              transpose<0, 1>(b(batch, reserve, row_index, range(begin, end)));

          prepared_layer_norm(task_context, active_act, left_norm_scale,
                              left_norm_offset, active_norm);
          prepared_projection(task_context, rows, C, C, active_norm, gate_a,
                              active_gate);
          prepared_projection(task_context, rows, C, C, active_norm, value_a,
                              active_value);
          using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
          using ValueTag = vec::Rebind<ComputeType, StorageTag>;
          using MaskTag = vec::Rebind<int8_t, ValueTag>;
          const nint_t lanes = vec::size(ValueTag{});
          const auto zero = vec::zeros(ValueTag{});
          const auto *mask_data = reinterpret_cast<const int8_t *>(
              batch_mask.data() + begin);
          for (nint_t channel = 0; channel < channels; ++channel) {
            const auto *gate_data = active_gate.data() +
                                    channel * active_gate.stride(1);
            const auto *value_data = active_value.data() +
                                     channel * active_value.stride(1);
            auto *output_a = active_a.data() + channel * active_a.stride(1);
            for (nint_t offset = 0; offset < rows; offset += lanes) {
              const nint_t active = std::min(
                  lanes, static_cast<nint_t>(rows) - offset);
              const auto gate_value = vec::load_convert(
                  ValueTag{}, gate_data + offset, vec::opt::first(active),
                  vec::opt::zero);
              const auto projected = vec::load_convert(
                  ValueTag{}, value_data + offset, vec::opt::first(active),
                  vec::opt::zero);
              auto result = vec::mul(
                  ValueTag{}, projected,
                  tail_safe_sigmoid_vec(ValueTag{}, gate_value));
              const auto mask_values = vec::load(
                  MaskTag{}, mask_data + offset, vec::opt::first(active),
                  vec::opt::zero);
              const auto mask_predicate = vec::convert(
                  ValueTag{}, MaskTag{},
                  vec::cmpne(MaskTag{}, mask_values,
                             vec::zeros(MaskTag{})));
              if (!keep_row)
                result = zero;
              else
                result = vec::blend(ValueTag{}, zero, mask_predicate, result);
              vec::store_convert(ValueTag{}, output_a + offset, result,
                                 vec::opt::first(active));
            }
          }
          prepared_projection(task_context, rows, C, C, active_norm, gate_b,
                              active_gate);
          prepared_projection(task_context, rows, C, C, active_norm, value_b,
                              active_value);
          for (nint_t channel = 0; channel < channels; ++channel) {
            const auto *gate_data = active_gate.data() +
                                    channel * active_gate.stride(1);
            const auto *value_data = active_value.data() +
                                     channel * active_value.stride(1);
            auto *output_b = active_b.data() + channel * active_b.stride(1);
            for (nint_t offset = 0; offset < rows; offset += lanes) {
              const nint_t active = std::min(
                  lanes, static_cast<nint_t>(rows) - offset);
              const auto gate_value = vec::load_convert(
                  ValueTag{}, gate_data + offset, vec::opt::first(active),
                  vec::opt::zero);
              const auto projected = vec::load_convert(
                  ValueTag{}, value_data + offset, vec::opt::first(active),
                  vec::opt::zero);
              auto result = vec::mul(
                  ValueTag{}, projected,
                  tail_safe_sigmoid_vec(ValueTag{}, gate_value));
              const auto mask_values = vec::load(
                  MaskTag{}, mask_data + offset, vec::opt::first(active),
                  vec::opt::zero);
              const auto mask_predicate = vec::convert(
                  ValueTag{}, MaskTag{},
                  vec::cmpne(MaskTag{}, mask_values,
                             vec::zeros(MaskTag{})));
              if (!keep_row)
                result = zero;
              else
                result = vec::blend(ValueTag{}, zero, mask_predicate, result);
              vec::store_convert(ValueTag{}, output_b + offset, result,
                                 vec::opt::first(active));
            }
          }
        });
  }
  projection_workspace.close();

  // Phase 2: independent channel contractions, [N,N] x [N,N]^T.
  const nint_t contraction_tasks = batch_count * channels;
  auto contraction_workspace = workspace.serial_scope("contraction");
  auto contraction_sample_a = a(0, 0, reserve, reserve);
  auto contraction_sample_b = b(0, 0, reserve, reserve);
  auto contraction_sample_product = product(0, 0, reserve, reserve);
  auto prepared_contraction = matmul_op.template prepare<Parallelism>(
      workspace, "contraction_scratch", N, N, N,
      tensor::unbind(contraction_sample_a),
      tensor::unbind(contraction_sample_b),
      tensor::unbind(contraction_sample_product));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, contraction_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / channels;
        const nint_t channel = task % channels;
        prepared_contraction(task_context, a(batch, channel, reserve, reserve),
                             b(batch, channel, reserve, reserve),
                             product(batch, channel, reserve, reserve));
      });
  contraction_workspace.close();

  // Phase 3: center norm and output projection. The gate projection writes
  // the final residual result through a coordinate-aware epilogue.
  const nint_t output_tasks = batch_count * sequence * row_tiles;
  auto output_workspace = workspace.serial_scope("output");
  const TileRows tile_rows{std::min(sequence, RowsPerTask)};
  const auto tile_shape = make_shape(tile_rows, C);
  auto centered_workers =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("centered",
                                                               tile_shape);
  auto projected_workers =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("projected",
                                                               tile_shape);
  auto sample_centered = centered_workers(0, reserve, reserve);
  auto sample_projected = projected_workers(0, reserve, reserve);
  auto sample_product =
      transpose<0, 1>(product(0, reserve, 0, range(0, tile_rows)));
  auto sample_norm = normalized(0, 0, range(0, tile_rows), reserve);
  auto sample_act = act(0, 0, range(0, tile_rows), reserve);
  auto sample_out = output(0, 0, range(0, tile_rows), reserve);
  auto sample_projected_view =
      sample_projected(range(0, tile_rows), reserve);
  auto final_out = tensor::output<ComputeType>(
      sample_out,
      gated_residual_transform(sample_projected_view, sample_act));
  auto prepared_center_norm = layer_norm.template prepare<Parallelism>(
      workspace, "center_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_product)),
      tensor::unbind(tensor::input<ComputeType>(center_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(center_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_centered)));
  auto prepared_output_projection = matmul_op.template prepare<Parallelism>(
      workspace, "output_projection_scratch", tile_rows, C, C,
      tensor::unbind(sample_centered), tensor::unbind(output_projection_b),
      tensor::unbind(sample_projected));
  auto prepared_gate = matmul_op.template prepare<Parallelism>(
      workspace, "gate_projection_scratch", tile_rows, C, C,
      tensor::unbind(sample_norm), tensor::unbind(gating_linear_b),
      tensor::unbind(final_out));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, output_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        auto centered = task_context.local(centered_workers);
        auto projected = task_context.local(projected_workers);
        const nint_t tile = task % row_tiles;
        const nint_t matrix = task / row_tiles;
        const nint_t batch = matrix / sequence;
        const nint_t row_index = matrix % sequence;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const TileRows rows{end - begin};
        auto active_product = transpose<0, 1>(
            product(batch, reserve, row_index, range(begin, end)));
        auto active_norm =
            normalized(batch, row_index, range(begin, end), reserve);
        auto active_act = act(batch, row_index, range(begin, end), reserve);
        auto active_out = output(batch, row_index, range(begin, end), reserve);
        auto active_centered = centered(range(0, rows), reserve);
        auto active_projected = projected(range(0, rows), reserve);

        prepared_center_norm(task_context, active_product, center_norm_scale,
                             center_norm_offset, active_centered);
        prepared_output_projection(task_context, rows, C, C, active_centered,
                                   output_projection_b, active_projected);
        prepared_gate(task_context, rows, C, C, active_norm, gating_linear_b,
                      tensor::output<ComputeType>(
                          active_out, gated_residual_transform(active_projected,
                                                               active_act)));
      });
  output_workspace.close();
  kernel_workspace.close();
}
