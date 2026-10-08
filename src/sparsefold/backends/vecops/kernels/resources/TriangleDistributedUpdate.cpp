// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file TriangleDistributedUpdate.cpp
 *  @brief Communication-back phase of distributed triangle multiplication.
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
#include "vecops/tensor/Transform.h"

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
#error "Distributed triangle update requires AMX or SME"
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

template <TensorLike Projected, TensorLike Residual>
auto gated_residual_transform(Projected projected, Residual residual) {
  const auto *projected_data = projected.data();
  const auto *residual_data = residual.data();
  const nint_t projected_stride = projected.stride(0);
  const nint_t residual_stride = residual.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [projected_data, residual_data, projected_stride, residual_stride](
          auto tag, auto gate, const auto &context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto projected_value = vec::load_convert(
            tag, projected_data + row * projected_stride + column,
            context.active_option(tag), vec::opt::zero);
        const auto residual_value = vec::load_convert(
            tag, residual_data + row * residual_stride + column,
            context.active_option(tag), vec::opt::zero);
#if defined(HAS_SME)
        const auto sigmoid =
            sparsefold_vecops::polynomial_sigmoid<ComputeType>(tag, gate);
#else
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(tag, gate);
#endif
        return vec::add(tag, residual_value,
                        vec::mul(tag, projected_value, sigmoid));
      });
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto normalized, TensorLike auto left,
                TensorLike auto right, TensorLike auto center_norm_scale,
                TensorLike auto center_norm_offset,
                TensorLike auto output_projection,
                TensorLike auto gating_projection, TensorLike auto output) {
  assert_shape(act, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(normalized, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(left, {vecops::any, C, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(right, {vecops::any, C, N, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(center_norm_scale, {C}, {cint<1>});
  assert_shape(center_norm_offset, {C}, {cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(gating_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});

  const nint_t batch_count = act.size(0);
  const LocalExtent local_extent{act.size(1)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t channels = static_cast<nint_t>(C);
  if (batch_count == 0 || local_rows == 0 || sequence == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const auto product_shape =
      make_shape(BatchExtent{batch_count}, C, local_extent, N);
  auto lifetime = workspace.serial_scope("triangle_distributed_update");
  auto product = workspace.tensor<vecops::bfloat16_t>("product", product_shape);

  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto output_b = packed_b(output_projection, 0, C, C);
  auto gate_b = packed_b(gating_projection, 0, C, C);

  const nint_t contraction_tasks = batch_count * channels;
  auto contraction_phase = workspace.serial_scope("contraction");
  auto prepared_contraction = matmul_op.template prepare<Parallelism>(
      workspace, "matmul_scratch", local_extent, N, N,
      tensor::unbind(left(0, 0, reserve, reserve)),
      tensor::unbind(right(0, 0, reserve, reserve)),
      tensor::unbind(product(0, 0, reserve, reserve)));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, contraction_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / channels;
        const nint_t channel = task % channels;
        prepared_contraction(task_context,
                             left(batch, channel, reserve, reserve),
                             right(batch, channel, reserve, reserve),
                             product(batch, channel, reserve, reserve));
      });
  contraction_phase.close();

  const nint_t tiles = ceil_div(sequence, RowsPerTask);
  const nint_t tasks = batch_count * local_rows * tiles;
  const TileRows sample_rows{std::min(sequence, RowsPerTask)};
  const auto tile_shape = make_shape(sample_rows, C);
  auto output_phase = workspace.serial_scope("output");
  auto centered_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("centered",
                                                               tile_shape);
  auto projected_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("projected",
                                                               tile_shape);
  const execution::TaskContext<Parallelism> sample_task{0};
  auto sample_centered_worker = sample_task.local(centered_storage);
  auto sample_projected_worker = sample_task.local(projected_storage);
  auto sample_centered = sample_centered_worker(range(0, sample_rows), reserve);
  auto sample_projected =
      sample_projected_worker(range(0, sample_rows), reserve);
  auto sample_product =
      transpose<0, 1>(product(0, reserve, 0, range(0, sample_rows)));
  auto sample_norm = normalized(0, 0, range(0, sample_rows), reserve);
  auto sample_act = act(0, 0, range(0, sample_rows), reserve);
  auto sample_out = output(0, 0, range(0, sample_rows), reserve);
  auto prepared_layer_norm = layer_norm.template prepare<Parallelism>(
      workspace, "layer_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_product)),
      tensor::unbind(tensor::input<ComputeType>(center_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(center_norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_centered)));
  auto prepared_projection = matmul_op.template prepare<Parallelism>(
      workspace, "output_projection_scratch", sample_rows, C, C,
      tensor::unbind(sample_centered), tensor::unbind(output_b),
      tensor::unbind(sample_projected));
  auto prepared_gate = matmul_op.template prepare<Parallelism>(
      workspace, "gating_projection_scratch", sample_rows, C, C,
      tensor::unbind(sample_norm), tensor::unbind(gate_b),
      tensor::unbind(tensor::output<ComputeType>(
          sample_out, gated_residual_transform(sample_projected, sample_act))));

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
        auto centered = task_context.local(centered_storage);
        auto projected = task_context.local(projected_storage);
        auto active_centered = centered(range(0, rows), reserve);
        auto active_projected = projected(range(0, rows), reserve);
        auto active_product = transpose<0, 1>(
            product(batch, reserve, local_row, range(begin, end)));
        auto active_norm =
            normalized(batch, local_row, range(begin, end), reserve);
        auto active_act = act(batch, local_row, range(begin, end), reserve);
        auto active_out = output(batch, local_row, range(begin, end), reserve);

        prepared_layer_norm(task_context, active_product, center_norm_scale,
                            center_norm_offset, active_centered);
        prepared_projection(task_context, rows, C, C, active_centered, output_b,
                            active_projected);
        prepared_gate(task_context, rows, C, C, active_norm, gate_b,
                      tensor::output<ComputeType>(
                          active_out, gated_residual_transform(active_projected,
                                                               active_act)));
      });
  output_phase.close();
  lifetime.close();
}
