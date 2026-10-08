// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file DiffusionTransitionBlock.cpp
 *  @brief Fused conditional transition block used by the diffusion transformer.
 */

#include <algorithm>
#include <tuple>
#include <type_traits>

#include "vecops/Kernel.h"
#include "vecops/execution/TaskPartition.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto M = spec::Hidden;
constexpr auto HiddenPanel = spec::HiddenPanel;
constexpr auto OutputPanel = spec::OutputPanel;
using BatchExtent = spec::B;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "DiffusionTransitionBlock requires an AMX or SME BF16 matmul target"
#endif

namespace {

#if defined(ARCH_X86_FAMILY)
constexpr nint_t RowsPerTask =
    decltype(C)::value >= 512
        ? (decltype(N)::value == 256 ? 24
                                     : (decltype(N)::value < 512 ? 16 : 32))
        : 48;
#else
// A 64-byte SME streaming vector naturally produces 32 FP32 rows.  Keeping
// full ZA tiles is more important than exposing extra tasks at short N.
constexpr nint_t RowsPerTask = 32;
#endif
constexpr nint_t RowAlignment =
    decltype(N)::value == 256 && RowsPerTask == 24 ? 8 : 1;
using ActiveRows = meta::Dynamic<RowAlignment, RowAlignment, RowsPerTask>;
using TileRows = std::conditional_t<(decltype(N)::value % RowsPerTask) == 0,
                                    meta::Const<RowsPerTask>, ActiveRows>;

// PyTorch materializes each BF16 projection and elementwise result. Preserve
// those rounding boundaries inside the fused epilogues: omitting them changes
// the denoising trajectory after hundreds of diffusion steps.
template <typename Tag, typename Value>
auto round_bf16(Tag tag, Value value) {
  using BF16Tag = vec::Rebind<vecops::bfloat16_t, Tag>;
  return vec::convert(tag, BF16Tag{},
                      vec::convert(BF16Tag{}, tag, value));
}

// Match the bounded CPU reference sigmoid without entering the variable
// latency vector-exp path. This also keeps saturated BF16 gates identical to
// the reference during the long denoising recurrence.
template <typename Tag, typename Value>
auto bounded_sigmoid(Tag tag, Value value) {
  const auto one = vec::fill(tag, 1.0f);
  const auto bounded = vec::max(
      tag, vec::fill(tag, -20.0f),
      vec::min(tag, value, vec::fill(tag, 20.0f)));
  const auto z = vec::mul(
      tag, vec::neg(tag, vec::abs(tag, bounded)),
      vec::fill(tag, 1.0f / 256.0f));
  auto polynomial = vec::add(
      tag, vec::fill(tag, 1.0f / 24.0f),
      vec::mul(tag, z, vec::fill(tag, 1.0f / 120.0f)));
  polynomial = vec::add(tag, vec::fill(tag, 1.0f / 6.0f),
                        vec::mul(tag, z, polynomial));
  polynomial = vec::add(tag, vec::fill(tag, 0.5f),
                        vec::mul(tag, z, polynomial));
  polynomial = vec::add(tag, one, vec::mul(tag, z, polynomial));
  auto exp_abs = vec::add(tag, one, vec::mul(tag, z, polynomial));
  for (int i = 0; i < 8; ++i)
    exp_abs = vec::mul(tag, exp_abs, exp_abs);
  const auto denominator = vec::add(tag, one, exp_abs);
  return vec::blend(
      tag, vec::div(tag, exp_abs, denominator),
      vec::cmpge(tag, bounded, vec::zeros(tag)),
      vec::div(tag, one, denominator));
}

auto silu_transform() {
  return make_elementwise_vec_transform<ComputeType, ComputeType>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        const auto projected = round_bf16(tag, value);
        const auto sigmoid = round_bf16(tag, bounded_sigmoid(tag, projected));
        return round_bf16(tag, vec::mul(tag, projected, sigmoid));
      });
}

template <TensorLike Gate> auto multiply_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            const auto coordinate = context.lane_coord(0);
            const nint_t row = coordinate[0];
            const nint_t column = coordinate[1];
            const auto other =
                vec::load_convert(tag, gate_data + row * gate_stride + column,
                                  context.active_option(tag), vec::opt::zero);
            return vec::mul(tag, round_bf16(tag, value), other);
          });
}

template <TensorLike Scale, TensorLike Bias>
auto adaptive_norm_transform(Scale scale, Bias bias) {
  const auto *scale_data = scale.data();
  const auto *bias_data = bias.data();
  const nint_t scale_stride = scale.stride(0);
  const nint_t bias_stride = bias.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [scale_data, bias_data, scale_stride, bias_stride](
          auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto active = context.active_option(tag);
        const auto scale_value =
            vec::load_convert(tag, scale_data + row * scale_stride + column,
                              active, vec::opt::zero);
        const auto bias_value =
            vec::load_convert(tag, bias_data + row * bias_stride + column,
                              active, vec::opt::zero);
        const auto sigmoid = round_bf16(tag, bounded_sigmoid(tag, scale_value));
        const auto normalized = round_bf16(tag, value);
        const auto scaled = round_bf16(
            tag, vec::mul(tag, sigmoid, normalized));
        return vec::add(tag, scaled, bias_value);
      });
}

template <TensorLike Gate, TensorLike Residual>
auto sigmoid_gate_residual_transform(Gate gate, Residual residual) {
  const auto *gate_data = gate.data();
  const auto *residual_data = residual.data();
  const nint_t gate_stride = gate.stride(0);
  const nint_t residual_stride = residual.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, residual_data, gate_stride, residual_stride](
          auto tag, auto value, const auto &context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto gate_value =
            vec::load_convert(tag, gate_data + row * gate_stride + column,
                              context.active_option(tag), vec::opt::zero);
        const auto sigmoid = round_bf16(tag, bounded_sigmoid(tag, gate_value));
        const auto gated = round_bf16(
            tag, vec::mul(tag, round_bf16(tag, value), sigmoid));
        const auto residual_value = vec::load_convert(
            tag, residual_data + row * residual_stride + column,
            context.active_option(tag), vec::opt::zero);
        return vec::add(tag, gated, residual_value);
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

template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b_panel(Packed packed, nint_t branch, nint_t panel, Spatial spatial,
                    Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(2) >= numel(layout), "packed B panel is too small");
  return make_tensor(packed.data() + branch * packed.stride(0) +
                         panel * packed.stride(1),
                     layout);
}

template <TensorLike Matrix, meta::ValueType Rows, meta::ValueType Columns,
          meta::ValueType RowStride>
auto matrix_panel(Matrix matrix, nint_t column_begin, Rows rows,
                  Columns columns, RowStride row_stride) {
  return make_tensor(matrix.data() + column_begin,
                     make_layout(make_shape(rows, columns),
                                 make_strides(row_stride, cint<1>)));
}

} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto act,
                TensorLike auto adaptive_scale, TensorLike auto adaptive_bias,
                TensorLike auto zero_gate, TensorLike auto transition1,
                TensorLike auto transition2, TensorLike auto output) {
  assert_shape(act, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(adaptive_scale, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(adaptive_bias, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(zero_gate, {vecops::any, N, C}, {vecops::any, C, cint<1>});
  assert_shape(transition1, {cint<2>, vecops::any, vecops::any},
               {vecops::any, vecops::any, cint<1>});
  assert_shape(transition2, {vecops::any, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, N, C}, {vecops::any, C, cint<1>});

  const nint_t batch_count = act.size(0);
  const nint_t cond_batch_count = adaptive_scale.size(0);
  const nint_t sequence = static_cast<nint_t>(N);
  VECOPS_ASSERT(batch_count % cond_batch_count == 0,
                "conditioning batch must divide the activation batch");
  if (batch_count == 0 || sequence == 0)
    return;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto pack_a_op =
      ops::matmul_pack(ops::MatmulPackConfig<MatmulAtom, matmul::Operand::A>{});
  constexpr auto TileCount = ceil_div(N, cint<RowsPerTask>);
  const nint_t tiles = static_cast<nint_t>(TileCount);
  const TileRows tile_rows{std::min(sequence, RowsPerTask)};
  const nint_t hidden = static_cast<nint_t>(M);
  const nint_t channels = static_cast<nint_t>(C);
  const nint_t hidden_panel = static_cast<nint_t>(HiddenPanel);
  const nint_t output_panel = static_cast<nint_t>(OutputPanel);
  VECOPS_ASSERT(hidden % hidden_panel == 0 && channels % output_panel == 0,
                "transition panel widths must divide logical dimensions");
  constexpr auto UpPanels = M / HiddenPanel;
  constexpr auto DownPanels = C / OutputPanel;
  const nint_t up_panels = static_cast<nint_t>(UpPanels);
  const nint_t down_panels = static_cast<nint_t>(DownPanels);
  VECOPS_ASSERT(transition1.size(1) == up_panels &&
                    transition2.size(0) == down_panels,
                "transition panel count does not match logical dimensions");
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  // Compare one worker-local pipeline per row tile with four panelized phases.
  // The task planner models incomplete worker waves and the extra fine-path
  // synchronization, avoiding fragile thresholds around lane-count multiples.
  const auto coarse_tasks = BatchExtent{batch_count} * TileCount;
  const auto coarse = execution::task_partition_candidate(
      cint<0>, execution::task_partition_phase(
                   coarse_tasks, cint<2> + cint<2> * UpPanels + DownPanels));
  const auto fine = execution::task_partition_candidate(
      cint<3>, execution::task_partition_phase(coarse_tasks, cint<2>),
      execution::task_partition_phase(coarse_tasks * UpPanels, cint<1>),
      execution::task_partition_phase(coarse_tasks * UpPanels, cint<1>),
      execution::task_partition_phase(coarse_tasks * DownPanels, cint<1>));
  const auto candidates = std::tuple{coarse, fine};
  const auto partition =
      execution::choose_task_partition(cint<Parallelism>, candidates);
  if (partition.candidate_index == cint<0>) {
    const auto local_normalized_shape = make_shape(tile_rows, C);
    const auto local_hidden_shape = make_shape(tile_rows, M);
    const auto packed_normalized_layout =
        matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
            make_layout(local_normalized_shape));
    auto coarse_scope = workspace.serial_scope("sample_row_tiles");
    auto normalized_storage =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "normalized", local_normalized_shape);
    auto hidden_storage =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "hidden", local_hidden_shape);
    auto packed_normalized_storage =
        workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
            "packed_normalized", packed_normalized_layout);
    const execution::TaskContext<Parallelism> sample_task{0};
    auto sample_normalized_worker = sample_task.local(normalized_storage);
    auto sample_hidden_worker = sample_task.local(hidden_storage);
    auto sample_normalized =
        sample_normalized_worker(range(0, tile_rows), reserve);
    auto sample_hidden = sample_hidden_worker(range(0, tile_rows), reserve);
    auto sample_packed_normalized =
        sample_task.local(packed_normalized_storage);
    auto sample_act = act(0, range(0, tile_rows), reserve);
    auto sample_scale = adaptive_scale(0, range(0, tile_rows), reserve);
    auto sample_bias = adaptive_bias(0, range(0, tile_rows), reserve);
    constexpr auto sample_up_columns = HiddenPanel;
    constexpr auto sample_down_columns = OutputPanel;
    auto sample_hidden_panel =
        matrix_panel(sample_hidden, 0, tile_rows, sample_up_columns, M);
    auto sample_output_full = output(0, range(0, tile_rows), reserve);
    auto sample_output_panel =
        matrix_panel(sample_output_full, 0, tile_rows, sample_down_columns, C);
    auto sample_zero_full = zero_gate(0, range(0, tile_rows), reserve);
    auto sample_act_full = act(0, range(0, tile_rows), reserve);
    auto prepared_norm = layer_norm.template prepare<Parallelism>(
        workspace, "norm_scratch",
        tensor::unbind(tensor::input<ComputeType>(sample_act)), tensor::nullopt,
        tensor::nullopt,
        tensor::unbind(tensor::output<ComputeType>(
            sample_normalized,
            adaptive_norm_transform(sample_scale, sample_bias))));
    auto prepared_pack = pack_a_op.template prepare<Parallelism>(
        workspace, "pack_scratch", tensor::unbind(sample_normalized),
        tensor::unbind(sample_packed_normalized));
    auto prepared_gate = matmul_op.template prepare<Parallelism>(
        workspace, "gate_scratch", tile_rows, sample_up_columns, C,
        tensor::unbind(sample_packed_normalized),
        tensor::unbind(packed_b_panel(transition1, 0, 0, sample_up_columns, C)),
        tensor::unbind(tensor::output<ComputeType>(sample_hidden_panel,
                                                   silu_transform())));
    auto prepared_value = matmul_op.template prepare<Parallelism>(
        workspace, "value_scratch", tile_rows, sample_up_columns, C,
        tensor::unbind(sample_packed_normalized),
        tensor::unbind(packed_b_panel(transition1, 1, 0, sample_up_columns, C)),
        tensor::unbind(tensor::output<ComputeType>(
            sample_hidden_panel, multiply_transform(sample_hidden_panel))));
    auto prepared_output = matmul_op.template prepare<Parallelism>(
        workspace, "output_scratch", tile_rows, sample_down_columns, M,
        tensor::unbind(sample_hidden),
        tensor::unbind(packed_b(transition2, 0, sample_down_columns, M)),
        tensor::unbind(tensor::output<ComputeType>(
            sample_output_panel,
            sigmoid_gate_residual_transform(
                matrix_panel(sample_zero_full, 0, tile_rows,
                             sample_down_columns, C),
                matrix_panel(sample_act_full, 0, tile_rows, sample_down_columns,
                             C)))));

    workspace.parallel_for<Parallelism>(
        nint_t{0}, static_cast<nint_t>(coarse_tasks), cint<1>,
        [&](execution::TaskContext<Parallelism> task_context,
            const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t task = work_item.begin();
          const nint_t batch_index = task / tiles;
          const nint_t tile = task % tiles;
          const nint_t row_begin = tile * RowsPerTask;
          const nint_t row_end = std::min(row_begin + RowsPerTask, sequence);
          const TileRows rows{row_end - row_begin};
          const nint_t cond_index = batch_index % cond_batch_count;
          auto normalized = task_context.local(normalized_storage);
          auto hidden_tile = task_context.local(hidden_storage);
          auto packed_storage = task_context.local(packed_normalized_storage);
          auto active_normalized = normalized(range(0, rows), reserve);
          auto active_hidden = hidden_tile(range(0, rows), reserve);
          prepared_norm(
              task_context,
              act(batch_index, range(row_begin, row_end), reserve),
              tensor::nullopt, tensor::nullopt,
              tensor::output<ComputeType>(
                  active_normalized,
                  adaptive_norm_transform(
                      adaptive_scale(cond_index, range(row_begin, row_end),
                                     reserve),
                      adaptive_bias(cond_index, range(row_begin, row_end),
                                    reserve))));
          const auto active_packed_layout =
              matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
                  active_normalized.layout());
          auto active_packed_normalized =
              make_tensor(packed_storage.data(), active_packed_layout);
          prepared_pack(task_context, active_normalized,
                        active_packed_normalized);
          if (up_panels == 1) {
            auto active_hidden_panel =
                matrix_panel(active_hidden, 0, rows, HiddenPanel, M);
            prepared_gate(task_context, rows, HiddenPanel, C,
                          active_packed_normalized,
                          packed_b_panel(transition1, 0, 0, HiddenPanel, C),
                          tensor::output<ComputeType>(active_hidden_panel,
                                                      silu_transform()));
            prepared_value(task_context, rows, HiddenPanel, C,
                           active_packed_normalized,
                           packed_b_panel(transition1, 1, 0, HiddenPanel, C),
                           tensor::output<ComputeType>(
                               active_hidden_panel,
                               multiply_transform(active_hidden_panel)));
          } else {
            for (nint_t panel = 0; panel < up_panels; ++panel) {
              const nint_t col_begin = panel * hidden_panel;
              constexpr auto columns = HiddenPanel;
              auto active_hidden_panel =
                  matrix_panel(active_hidden, col_begin, rows, columns, M);
              prepared_gate(task_context, rows, columns, C,
                            active_packed_normalized,
                            packed_b_panel(transition1, 0, panel, columns, C),
                            tensor::output<ComputeType>(active_hidden_panel,
                                                        silu_transform()));
            }
            for (nint_t panel = 0; panel < up_panels; ++panel) {
              const nint_t col_begin = panel * hidden_panel;
              constexpr auto columns = HiddenPanel;
              auto active_hidden_panel =
                  matrix_panel(active_hidden, col_begin, rows, columns, M);
              prepared_value(task_context, rows, columns, C,
                             active_packed_normalized,
                             packed_b_panel(transition1, 1, panel, columns, C),
                             tensor::output<ComputeType>(
                                 active_hidden_panel,
                                 multiply_transform(active_hidden_panel)));
            }
          }
          if (down_panels == 1) {
            auto active_output_full =
                output(batch_index, range(row_begin, row_end), reserve);
            auto active_zero_full =
                zero_gate(cond_index, range(row_begin, row_end), reserve);
            auto active_act_full =
                act(batch_index, range(row_begin, row_end), reserve);
            prepared_output(
                task_context, rows, OutputPanel, M, active_hidden,
                packed_b(transition2, 0, OutputPanel, M),
                tensor::output<ComputeType>(
                    matrix_panel(active_output_full, 0, rows, OutputPanel, C),
                    sigmoid_gate_residual_transform(
                        matrix_panel(active_zero_full, 0, rows, OutputPanel, C),
                        matrix_panel(active_act_full, 0, rows, OutputPanel,
                                     C))));
          } else {
            for (nint_t panel = 0; panel < down_panels; ++panel) {
              const nint_t col_begin = panel * output_panel;
              constexpr auto columns = OutputPanel;
              auto active_output_full =
                  output(batch_index, range(row_begin, row_end), reserve);
              auto active_zero_full =
                  zero_gate(cond_index, range(row_begin, row_end), reserve);
              auto active_act_full =
                  act(batch_index, range(row_begin, row_end), reserve);
              prepared_output(task_context, rows, columns, M, active_hidden,
                              packed_b(transition2, panel, columns, M),
                              tensor::output<ComputeType>(
                                  matrix_panel(active_output_full, col_begin,
                                               rows, columns, C),
                                  sigmoid_gate_residual_transform(
                                      matrix_panel(active_zero_full, col_begin,
                                                   rows, columns, C),
                                      matrix_panel(active_act_full, col_begin,
                                                   rows, columns, C))));
            }
          }
        });
    coarse_scope.close();
    return;
  }

  // The intermediate buffers are global deliberately: barriers between the
  // four domains let each GEMM be partitioned over token tiles and output
  // panels rather than assigning an entire wide FFN to one worker.
  const auto normalized_shape = make_shape(BatchExtent{batch_count}, N, C);
  const auto hidden_shape = make_shape(BatchExtent{batch_count}, N, M);
  auto global_scope = workspace.serial_scope("global");
  auto normalized = workspace.tensor<vecops::bfloat16_t>(
      "normalized", normalized_shape,
      {.placement = execution::WorkspacePlacementPolicy::FastPreferred});
  auto hidden_act = workspace.tensor<vecops::bfloat16_t>(
      "hidden", hidden_shape,
      {.placement = execution::WorkspacePlacementPolicy::FastPreferred});

  // Domain 1: adaptive layer normalization, [B, tile(N)].
  const nint_t norm_tasks = batch_count * tiles;
  auto norm_scope = workspace.serial_scope("normalization");
  auto sample_act = act(0, range(0, tile_rows), reserve);
  auto sample_adaptive_scale = adaptive_scale(0, range(0, tile_rows), reserve);
  auto sample_adaptive_bias = adaptive_bias(0, range(0, tile_rows), reserve);
  auto sample_normalized = normalized(0, range(0, tile_rows), reserve);
  auto prepared_norm = layer_norm.template prepare<Parallelism>(
      workspace, "scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_act)), tensor::nullopt,
      tensor::nullopt,
      tensor::unbind(tensor::output<ComputeType>(
          sample_normalized, adaptive_norm_transform(sample_adaptive_scale,
                                                     sample_adaptive_bias))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, norm_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch_index = task / tiles;
        const nint_t tile = task % tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, sequence);
        const nint_t cond_index = batch_index % cond_batch_count;
        prepared_norm(
            task_context, act(batch_index, range(begin, end), reserve),
            tensor::nullopt, tensor::nullopt,
            tensor::output<ComputeType>(
                normalized(batch_index, range(begin, end), reserve),
                adaptive_norm_transform(
                    adaptive_scale(cond_index, range(begin, end), reserve),
                    adaptive_bias(cond_index, range(begin, end), reserve))));
      });
  norm_scope.close();

  const nint_t up_tasks = batch_count * tiles * up_panels;
  constexpr auto sample_up_columns = HiddenPanel;
  auto sample_hidden_full = hidden_act(0, range(0, tile_rows), reserve);
  auto sample_hidden =
      matrix_panel(sample_hidden_full, 0, tile_rows, sample_up_columns, M);

  // Domain 2: gate/SiLU projection, [B, tile(N), panel(M)].
  auto gate_scope = workspace.serial_scope("gate_projection");
  auto prepared_gate = matmul_op.template prepare<Parallelism>(
      workspace, "scratch", tile_rows, sample_up_columns, C,
      tensor::unbind(sample_normalized),
      tensor::unbind(packed_b_panel(transition1, 0, 0, sample_up_columns, C)),
      tensor::unbind(
          tensor::output<ComputeType>(sample_hidden, silu_transform())));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, up_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t panel = task % up_panels;
        const nint_t batch_tile = task / up_panels;
        const nint_t batch_index = batch_tile / tiles;
        const nint_t tile = batch_tile % tiles;
        const nint_t row_begin = tile * RowsPerTask;
        const nint_t row_end = std::min(row_begin + RowsPerTask, sequence);
        const nint_t col_begin = panel * hidden_panel;
        const TileRows rows{row_end - row_begin};
        constexpr auto columns = HiddenPanel;
        auto active_hidden_full =
            hidden_act(batch_index, range(row_begin, row_end), reserve);
        prepared_gate(
            task_context, rows, columns, C,
            normalized(batch_index, range(row_begin, row_end), reserve),
            packed_b_panel(transition1, 0, panel, columns, C),
            tensor::output<ComputeType>(
                matrix_panel(active_hidden_full, col_begin, rows, columns, M),
                silu_transform()));
      });
  gate_scope.close();

  // Domain 3: value projection and GLU, [B, tile(N), panel(M)].
  auto value_scope = workspace.serial_scope("value_projection");
  auto prepared_value = matmul_op.template prepare<Parallelism>(
      workspace, "scratch", tile_rows, sample_up_columns, C,
      tensor::unbind(sample_normalized),
      tensor::unbind(packed_b_panel(transition1, 1, 0, sample_up_columns, C)),
      tensor::unbind(tensor::output<ComputeType>(
          sample_hidden, multiply_transform(sample_hidden))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, up_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t panel = task % up_panels;
        const nint_t batch_tile = task / up_panels;
        const nint_t batch_index = batch_tile / tiles;
        const nint_t tile = batch_tile % tiles;
        const nint_t row_begin = tile * RowsPerTask;
        const nint_t row_end = std::min(row_begin + RowsPerTask, sequence);
        const nint_t col_begin = panel * hidden_panel;
        const TileRows rows{row_end - row_begin};
        constexpr auto columns = HiddenPanel;
        auto active_hidden_full =
            hidden_act(batch_index, range(row_begin, row_end), reserve);
        auto active_hidden =
            matrix_panel(active_hidden_full, col_begin, rows, columns, M);
        prepared_value(
            task_context, rows, columns, C,
            normalized(batch_index, range(row_begin, row_end), reserve),
            packed_b_panel(transition1, 1, panel, columns, C),
            tensor::output<ComputeType>(active_hidden,
                                        multiply_transform(active_hidden)));
      });
  value_scope.close();

  // Domain 4: output projection, [B, tile(N), panel(C)].
  const nint_t down_tasks = batch_count * tiles * down_panels;
  constexpr auto sample_down_columns = OutputPanel;
  auto sample_output_full = output(0, range(0, tile_rows), reserve);
  auto sample_output =
      matrix_panel(sample_output_full, 0, tile_rows, sample_down_columns, C);
  auto sample_zero_full = zero_gate(0, range(0, tile_rows), reserve);
  auto sample_zero_gate =
      matrix_panel(sample_zero_full, 0, tile_rows, sample_down_columns, C);
  auto sample_residual_full = act(0, range(0, tile_rows), reserve);
  auto sample_residual =
      matrix_panel(sample_residual_full, 0, tile_rows, sample_down_columns, C);
  auto down_scope = workspace.serial_scope("output_projection");
  auto prepared_output = matmul_op.template prepare<Parallelism>(
      workspace, "scratch", tile_rows, sample_down_columns, M,
      tensor::unbind(sample_hidden_full),
      tensor::unbind(packed_b(transition2, 0, sample_down_columns, M)),
      tensor::unbind(tensor::output<ComputeType>(
          sample_output,
          sigmoid_gate_residual_transform(sample_zero_gate, sample_residual))));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, down_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t panel = task % down_panels;
        const nint_t batch_tile = task / down_panels;
        const nint_t batch_index = batch_tile / tiles;
        const nint_t tile = batch_tile % tiles;
        const nint_t row_begin = tile * RowsPerTask;
        const nint_t row_end = std::min(row_begin + RowsPerTask, sequence);
        const nint_t col_begin = panel * output_panel;
        const TileRows rows{row_end - row_begin};
        constexpr auto columns = OutputPanel;
        const nint_t cond_index = batch_index % cond_batch_count;
        auto active_output_full =
            output(batch_index, range(row_begin, row_end), reserve);
        auto active_zero_full =
            zero_gate(cond_index, range(row_begin, row_end), reserve);
        auto active_act_full =
            act(batch_index, range(row_begin, row_end), reserve);
        prepared_output(
            task_context, rows, columns, M,
            hidden_act(batch_index, range(row_begin, row_end), reserve),
            packed_b(transition2, panel, columns, M),
            tensor::output<ComputeType>(
                matrix_panel(active_output_full, col_begin, rows, columns, C),
                sigmoid_gate_residual_transform(
                    matrix_panel(active_zero_full, col_begin, rows, columns, C),
                    matrix_panel(active_act_full, col_begin, rows, columns,
                                 C))));
      });
  down_scope.close();
  global_scope.close();
}
