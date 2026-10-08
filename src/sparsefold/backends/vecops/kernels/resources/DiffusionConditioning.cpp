// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file DiffusionConditioning.cpp
 *  @brief Precompute shared AdaLN conditioning for diffusion residual blocks.
 */

#include <algorithm>
#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

constexpr auto L = spec::L;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto S = spec::S;
using CondBatch = spec::BCond;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "DiffusionConditioning requires an AMX or SME BF16 matmul target"
#endif

namespace {

#if defined(ARCH_X86_FAMILY)
constexpr nint_t RowsPerTask = decltype(N)::value <= 256 ? 16 : 32;
#else
constexpr nint_t RowsPerTask = 32;
#endif

using ActiveRows = meta::Dynamic<1, 1, RowsPerTask>;
using TileRows = std::conditional_t<(decltype(N)::value % RowsPerTask) == 0,
                                    meta::Const<RowsPerTask>, ActiveRows>;

template <TensorLike Bias> auto bf16_projection_bias(Bias bias) {
  const auto *bias_data = bias.data();
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [bias_data](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
        using BF16Tag = vec::Rebind<vecops::bfloat16_t, decltype(tag)>;
        const auto projected = vec::convert(
            tag, BF16Tag{}, vec::convert(BF16Tag{}, tag, value));
        const auto column = context.lane_coord(0)[1];
        const auto offset = vec::load_convert(
            tag, bias_data + column, context.active_option(tag),
            vec::opt::zero);
        return vec::add(tag, projected, offset);
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

void __kernel__(execution::WorkspaceContext &workspace,
                TensorLike auto single_cond, TensorLike auto cond_norm_scale,
                TensorLike auto cond_scale_projection,
                TensorLike auto cond_scale_bias,
                TensorLike auto cond_bias_projection,
                TensorLike auto zero_cond_projection,
                TensorLike auto zero_cond_bias, TensorLike auto adaptive_scale,
                TensorLike auto adaptive_bias, TensorLike auto zero_gate) {
  assert_shape(single_cond, {vecops::any, N, S}, {vecops::any, S, cint<1>});
  assert_shape(cond_norm_scale, {L, S}, {S, cint<1>});
  assert_shape(cond_scale_projection, {L, vecops::any}, {vecops::any, cint<1>});
  assert_shape(cond_scale_bias, {L, C}, {C, cint<1>});
  assert_shape(cond_bias_projection, {L, vecops::any}, {vecops::any, cint<1>});
  assert_shape(zero_cond_projection, {L, vecops::any}, {vecops::any, cint<1>});
  assert_shape(zero_cond_bias, {L, C}, {C, cint<1>});
  assert_shape(adaptive_scale, {L, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(adaptive_bias, {L, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(zero_gate, {L, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});

  const nint_t layers = static_cast<nint_t>(L);
  const nint_t cond_batches = single_cond.size(0);
  const nint_t sequence = static_cast<nint_t>(N);
  if (layers == 0 || cond_batches == 0 || sequence == 0)
    return;

  const nint_t tiles = ceil_div(sequence, RowsPerTask);
  const TileRows tile_rows{std::min(sequence, RowsPerTask)};
  const auto cond_shape = make_shape(tile_rows, S);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const auto normalized_shape = make_shape(L, CondBatch{cond_batches}, N, S);
  auto global_scope = workspace.serial_scope("normalized_global");
  auto normalized = workspace.tensor<vecops::bfloat16_t>(
      "normalized", normalized_shape,
      {.placement = execution::WorkspacePlacementPolicy::FastPreferred});

  const nint_t norm_tasks = layers * cond_batches * tiles;
  auto norm_scope = workspace.serial_scope("normalize");
  auto sample_cond = single_cond(0, range(0, tile_rows), reserve);
  auto sample_norm = normalized(0, 0, range(0, tile_rows), reserve);
  auto prepared_norm = layer_norm.template prepare<Parallelism>(
      workspace, "scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_cond)),
      tensor::unbind(
          tensor::input<ComputeType>(cond_norm_scale(0, reserve))),
      tensor::unbind(tensor::output<ComputeType>(sample_norm)));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, norm_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t tile = task % tiles;
        const nint_t batch_layer = task / tiles;
        const nint_t cond_index = batch_layer % cond_batches;
        const nint_t layer = batch_layer / cond_batches;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        prepared_norm(
            task_context,
            single_cond(cond_index, range(begin, end), reserve),
            cond_norm_scale(layer, reserve),
            normalized(layer, cond_index, range(begin, end), reserve));
      });
  norm_scope.close();

  constexpr nint_t Projections = 3;
  const nint_t projection_tasks = norm_tasks * Projections;
  auto projection_scope = workspace.serial_scope("projection");
  auto sample_output = adaptive_scale(0, 0, range(0, tile_rows), reserve);
  auto sample_bias_matrix = make_tensor(
      cond_scale_bias.data(),
      make_layout(make_shape(tile_rows, C), make_strides(cint<0>, cint<1>)));
  auto sample_packed = packed_b(cond_scale_projection, 0, C, S);
  auto prepared_scale = matmul_op.template prepare<Parallelism>(
      workspace, "scale_scratch", tile_rows, C, S,
      tensor::unbind(sample_norm), tensor::unbind(sample_packed),
      tensor::unbind(tensor::output<ComputeType>(
          sample_output, bf16_projection_bias(sample_bias_matrix))));
  auto prepared_bias = matmul_op.template prepare<Parallelism>(
      workspace, "bias_scratch", tile_rows, C, S,
      tensor::unbind(sample_norm), tensor::unbind(sample_packed),
      tensor::unbind(adaptive_bias(0, 0, range(0, tile_rows), reserve)));
  auto prepared_zero = matmul_op.template prepare<Parallelism>(
      workspace, "zero_scratch", tile_rows, C, S,
      tensor::unbind(sample_cond), tensor::unbind(sample_packed),
      tensor::unbind(tensor::output<ComputeType>(
          zero_gate(0, 0, range(0, tile_rows), reserve),
          bf16_projection_bias(sample_bias_matrix))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, projection_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t projection = task % Projections;
        const nint_t norm_task = task / Projections;
        const nint_t tile = norm_task % tiles;
        const nint_t batch_layer = norm_task / tiles;
        const nint_t cond_index = batch_layer % cond_batches;
        const nint_t layer = batch_layer / cond_batches;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const TileRows rows{end - begin};
        auto scale_bias = make_tensor(
            cond_scale_bias.data() + layer * cond_scale_bias.stride(0),
            make_layout(make_shape(rows, C),
                        make_strides(cint<0>, cint<1>)));
        auto zero_bias = make_tensor(
            zero_cond_bias.data() + layer * zero_cond_bias.stride(0),
            make_layout(make_shape(rows, C),
                        make_strides(cint<0>, cint<1>)));
        if (projection == 0) {
          prepared_scale(
              task_context, rows, C, S,
              normalized(layer, cond_index, range(begin, end), reserve),
              packed_b(cond_scale_projection, layer, C, S),
              tensor::output<ComputeType>(
                  adaptive_scale(layer, cond_index, range(begin, end), reserve),
                  bf16_projection_bias(scale_bias)));
        } else if (projection == 1) {
          prepared_bias(
              task_context, rows, C, S,
              normalized(layer, cond_index, range(begin, end), reserve),
              packed_b(cond_bias_projection, layer, C, S),
              adaptive_bias(layer, cond_index, range(begin, end), reserve));
        } else {
          prepared_zero(
              task_context, rows, C, S,
              single_cond(cond_index, range(begin, end), reserve),
              packed_b(zero_cond_projection, layer, C, S),
              tensor::output<ComputeType>(
                  zero_gate(layer, cond_index, range(begin, end), reserve),
                  bf16_projection_bias(zero_bias)));
        }
      });
  projection_scope.close();
  global_scope.close();
}
