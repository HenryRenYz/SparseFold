// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file TriangleDistributedProjection.cpp
 *  @brief Communication-front phase of distributed triangle multiplication.
 */

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
using LocalExtent = spec::L;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed triangle projection requires AMX or SME"
#endif

namespace {

constexpr nint_t RowsPerTask = 128;
constexpr nint_t MaxTileRows =
    decltype(N)::value < RowsPerTask ? decltype(N)::value : RowsPerTask;
using ActiveRows = meta::Dynamic<1, 1, MaxTileRows>;
using TileRows = std::conditional_t<(decltype(N)::value <= RowsPerTask) ||
                                        (decltype(N)::value % RowsPerTask == 0),
                                    meta::Const<MaxTileRows>, ActiveRows>;

template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, nint_t index, Spatial spatial,
              Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data() + index * packed.stride(0), layout);
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto row_mask, TensorLike auto column_mask,
                TensorLike auto norm_scale, TensorLike auto norm_offset,
                TensorLike auto packed_projection,
                TensorLike auto normalized, TensorLike auto a,
                TensorLike auto b) {
  assert_shape(act, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(row_mask, {vecops::any, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(column_mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(norm_scale, {C}, {cint<1>});
  assert_shape(norm_offset, {C}, {cint<1>});
  assert_shape(packed_projection, {cint<4>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(normalized, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(a, {vecops::any, C, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(b, {vecops::any, C, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});

  const nint_t batch_count = act.size(0);
  const LocalExtent local_extent{act.size(1)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t channels = static_cast<nint_t>(C);
  if (batch_count == 0 || local_rows == 0 || sequence == 0)
    return;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto gate_a = packed_b(packed_projection, 0, C, C);
  auto value_a = packed_b(packed_projection, 1, C, C);
  auto gate_b = packed_b(packed_projection, 2, C, C);
  auto value_b = packed_b(packed_projection, 3, C, C);

  const nint_t tiles = ceil_div(sequence, RowsPerTask);
  const nint_t tasks = batch_count * local_rows * tiles;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  const TileRows sample_rows{std::min(sequence, RowsPerTask)};
  // Keep the temporary projections channel-major.  The final A/B tensors are
  // also channel-major, so the sigmoid/mask epilogue can vectorize across the
  // contiguous row dimension without a strided scatter.
  const auto tile_shape = make_shape(C, sample_rows);
  auto phase = workspace.serial_scope("triangle_projection");
  auto gate_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("gate", tile_shape);
  auto value_workers = workspace.worker_tensor<
      vecops::bfloat16_t, Parallelism>("value", tile_shape);
  auto sample_act = act(0, 0, range(0, sample_rows), reserve);
  auto sample_norm = normalized(0, 0, range(0, sample_rows), reserve);
  auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_act)),
      tensor::unbind(tensor::input<ComputeType>(norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_norm)));
  auto prepared_matmul = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", sample_rows, C, C,
      tensor::unbind(sample_norm), tensor::unbind(gate_a),
      tensor::unbind(transpose<0, 1>(gate_workers(0, reserve, reserve))));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t tile = task % tiles;
        const nint_t matrix = task / tiles;
        const nint_t batch = matrix / local_rows;
        const nint_t local_row = matrix % local_rows;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const TileRows rows{end - begin};
        auto active_norm =
            normalized(batch, local_row, range(begin, end), reserve);
        auto gate = task_context.local(gate_workers);
        auto value = task_context.local(value_workers);
        auto active_gate =
            transpose<0, 1>(gate(reserve, range(0, rows)));
        auto active_value =
            transpose<0, 1>(value(reserve, range(0, rows)));
        auto active_a = transpose<0, 1>(
            a(batch, reserve, local_row, range(begin, end)));
        auto active_b = transpose<0, 1>(
            b(batch, reserve, local_row, range(begin, end)));

        prepared_layer_norm(
            task_context,
            act(batch, local_row, range(begin, end), reserve),
            norm_scale, norm_offset, active_norm);

        const auto project = [&](auto packed_gate, auto packed_value,
                                 auto destination)
            VECOPS_INLINE_LAMBDA_NOEXCEPT {
          prepared_matmul(
              task_context, active_norm, packed_gate, active_gate);
          prepared_matmul(
              task_context, active_norm, packed_value, active_value);
          const bool keep_local_row = row_mask(batch, local_row);
          using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
          using ValueTag = vec::Rebind<ComputeType, StorageTag>;
          using MaskTag = vec::Rebind<int8_t, ValueTag>;
          const nint_t lanes = vec::size(ValueTag{});
          const auto zero = vec::zeros(ValueTag{});
          const auto *mask_data = reinterpret_cast<const int8_t *>(
              column_mask.data() + batch * column_mask.stride(0) + begin);
          for (nint_t channel = 0; channel < channels; ++channel) {
            const auto *gate_data = active_gate.data() +
                                    channel * active_gate.stride(1);
            const auto *value_data = active_value.data() +
                                     channel * active_value.stride(1);
            auto *destination_data = destination.data() +
                                     channel * destination.stride(1);
            for (nint_t offset = 0; offset < rows; offset += lanes) {
              const nint_t active = std::min(
                  lanes, static_cast<nint_t>(rows) - offset);
              auto result = zero;
              if (keep_local_row) {
                const auto gate_value = vec::load_convert(
                    ValueTag{}, gate_data + offset,
                    vec::opt::first(active), vec::opt::zero);
                const auto projected = vec::load_convert(
                    ValueTag{}, value_data + offset,
                    vec::opt::first(active), vec::opt::zero);
#if defined(HAS_SME)
                const auto sigmoid =
                    sparsefold_vecops::polynomial_sigmoid<ComputeType>(
                        ValueTag{}, gate_value);
#else
                const auto sigmoid =
                    sparsefold_vecops::stable_sigmoid<ComputeType>(
                        ValueTag{}, gate_value);
#endif
                result = vec::mul(ValueTag{}, projected, sigmoid);
                const auto mask_values = vec::load(
                    MaskTag{}, mask_data + offset,
                    vec::opt::first(active), vec::opt::zero);
                const auto mask_predicate = vec::convert(
                    ValueTag{}, MaskTag{},
                    vec::cmpne(MaskTag{}, mask_values,
                               vec::zeros(MaskTag{})));
                result = vec::blend(
                    ValueTag{}, zero, mask_predicate, result);
              }
              vec::store_convert(
                  ValueTag{}, destination_data + offset, result,
                  vec::opt::first(active));
            }
          }
        };

        project(gate_a, value_a, active_a);
        project(gate_b, value_b, active_b);
      });
  phase.close();
}
