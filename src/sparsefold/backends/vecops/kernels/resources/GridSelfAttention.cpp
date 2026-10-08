// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file GridSelfAttention.cpp @brief AlphaFold 3 IBS grid self-attention. */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Attention.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::HeadDim;
constexpr auto T = spec::T;
constexpr auto S = spec::S;
constexpr bool Transpose = spec::Transpose;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "GridSelfAttention requires an AMX or SME BF16 matmul target"
#endif

namespace {

using AttentionConfig =
    ops::AttentionConfig<ops::MatmulConfig<MatmulAtom>, 32, 32,
                         ops::AttentionCausalMode::none,
                         ops::SDPAStrategy::automatic, vec::Accuracy::Estimate>;

template <TensorLike Gate> auto sigmoid_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value,
                               const auto &context) VECOPS_KERNEL_LAMBDA {
        VECOPS_ASSERT(context.vector_axis == 1,
                      "grid attention gate epilogue must traverse channels");
        const auto coordinate = context.lane_coord(0);
        const nint_t row = coordinate[0];
        const nint_t column = coordinate[1];
        const auto gate_value =
            vec::load_convert(tag, gate_data + row * gate_stride + column,
                              context.active_option(tag), vec::opt::zero);
#if defined(HAS_SME)
        const auto sigmoid = sparsefold_vecops::polynomial_sigmoid<ComputeType>(
            tag, gate_value);
#else
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
            tag, gate_value);
#endif
        return vec::mul(tag, value, sigmoid);
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
                TensorLike auto key_padding_mask, TensorLike auto q_projection,
                TensorLike auto k_projection, TensorLike auto v_projection,
                TensorLike auto output_projection, TensorLike auto g_projection,
                TensorLike auto bias_projection, TensorLike auto norm_scale,
                TensorLike auto norm_offset, TensorLike auto index_map,
                TensorLike auto weight_map, TensorLike auto output,
                int64_t random_blocks, float32_t static_probability,
                float32_t random_probability, float32_t minimum_probability,
                bool generate_maps, int64_t seed) {
  assert_shape(act, {vecops::any, N, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(key_padding_mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(q_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(k_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(v_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(g_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(bias_projection, {cint<1>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(norm_scale, {C}, {cint<1>});
  assert_shape(norm_offset, {C}, {cint<1>});
  assert_shape(index_map, {vecops::any, N, H, T, S},
               {vecops::any, vecops::any, vecops::any, S, cint<1>});
  assert_shape(weight_map, {vecops::any, N, H, T, S},
               {vecops::any, vecops::any, vecops::any, S, cint<1>});
  assert_shape(output, {vecops::any, N, N, C},
               {vecops::any, vecops::any, C, cint<1>});

  const auto batch_extent = BatchExtent{act.size(0)};
  const nint_t batch_count = act.size(0);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t channels = static_cast<nint_t>(C);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  const nint_t selected_blocks = static_cast<nint_t>(S);
  if (batch_count == 0 || sequence == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  AttentionConfig attention_config{};
  attention_config.selected_blocks = selected_blocks;
  attention_config.random_blocks = static_cast<nint_t>(random_blocks);
  attention_config.static_probability = static_probability;
  attention_config.random_probability = random_probability;
  attention_config.minimum_probability = minimum_probability;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto indexer = ops::ibs_attention_indexer(attention_config);
  auto attention_op = ops::ibs_attention(attention_config);
  auto bias_projection_b = packed_b(bias_projection, 0, H, C);
  auto output_projection_b = packed_b(output_projection, 0, C, H * D);
  const ComputeType attention_scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  const auto normalized_shape = make_shape(batch_extent, N, N, C);
  const auto bias_shape = make_shape(batch_extent, H, N, N);
  auto global_scope = workspace.serial_scope("global");
  auto normalized =
      workspace.tensor<vecops::bfloat16_t>("normalized", normalized_shape);
  auto bias = workspace.tensor<vecops::bfloat16_t>("bias", bias_shape);

  // The pair bias is always derived from the original orientation. Only the
  // attention axes are transposed for the incoming grid-attention variant.
  const nint_t normalization_tasks = batch_count * sequence;
  auto normalization_scope = workspace.serial_scope("normalization");
  auto sample_act = act(0, 0, reserve, reserve);
  auto sample_norm = normalized(0, 0, reserve, reserve);
  auto sample_bias = transpose<0, 1>(bias(0, reserve, 0, reserve));
  auto prepared_norm = layer_norm.template prepare<Parallelism>(
      workspace, "norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_act)),
      tensor::unbind(tensor::input<ComputeType>(norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(norm_offset)),
      tensor::unbind(tensor::output<ComputeType>(sample_norm)));
  auto prepared_bias = matmul_op.template prepare<Parallelism>(
      workspace, "bias_scratch", N, H, C, tensor::unbind(sample_norm),
      tensor::unbind(bias_projection_b), tensor::unbind(sample_bias));
  workspace.parallel_for<Parallelism>(
      nint_t{0}, normalization_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / sequence;
        const nint_t row = task % sequence;
        auto active_act = act(batch, row, reserve, reserve);
        auto active_norm = normalized(batch, row, reserve, reserve);
        auto active_bias = transpose<0, 1>(bias(batch, reserve, row, reserve));
        prepared_norm(task_context, active_act, norm_scale, norm_offset,
                      active_norm);
        prepared_bias(task_context, active_norm, bias_projection_b,
                      active_bias);
      });
  normalization_scope.close();

  const nint_t attention_tasks = batch_count * sequence;
  auto attention_scope = workspace.serial_scope("attention");
  const auto head_matrix_shape = make_shape(N, D);
  const auto weighted_shape = make_shape(N, H, D);
  auto q_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "q", head_matrix_shape);
  auto k_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "k", head_matrix_shape);
  auto v_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "v", head_matrix_shape);
  auto gate_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "gate", head_matrix_shape);
  auto weighted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("weighted",
                                                               weighted_shape);
  auto sample_q = q_values(0, reserve, reserve);
  auto sample_k = k_values(0, reserve, reserve);
  auto sample_v = v_values(0, reserve, reserve);
  auto sample_gate = gate_values(0, reserve, reserve);
  auto sample_weighted = weighted_values(0, reserve, reserve, reserve);
  auto sample_key_mask = key_padding_mask(0, reserve);
  auto sample_query_mask =
      make_tensor(key_padding_mask.data(),
                  make_layout(make_shape(N), make_strides(cint<0>)));
  auto sample_attention_bias = bias(0, 0, reserve, reserve);
  auto sample_index = index_map(0, 0, 0, reserve, reserve);
  auto sample_weight = weight_map(0, 0, 0, reserve, reserve);
  auto prepared_indexer = indexer.template prepare<Parallelism>(
      workspace, "index_scratch", tensor::unbind(sample_q),
      tensor::unbind(sample_k), tensor::unbind(sample_query_mask),
      tensor::unbind(sample_key_mask), tensor::nullopt,
      tensor::unbind(sample_attention_bias), tensor::unbind(sample_index),
      tensor::unbind(sample_weight));
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "attention_scratch", tensor::unbind(sample_q),
      tensor::unbind(sample_k), tensor::unbind(sample_v),
      tensor::unbind(sample_index), tensor::unbind(sample_weight),
      tensor::unbind(sample_query_mask), tensor::unbind(sample_key_mask),
      tensor::nullopt, tensor::unbind(sample_attention_bias),
      tensor::unbind(
          tensor::output<ComputeType>(sample_weighted(reserve, 0, reserve),
                                      sigmoid_gate_transform(sample_gate))));
  auto attention_norm_pattern = [&] {
    if constexpr (Transpose)
      return normalized(0, reserve, 0, reserve);
    else
      return normalized(0, 0, reserve, reserve);
  }();
  auto attention_residual_pattern = [&] {
    if constexpr (Transpose)
      return act(0, reserve, 0, reserve);
    else
      return act(0, 0, reserve, reserve);
  }();
  auto attention_output_pattern = [&] {
    if constexpr (Transpose)
      return output(0, reserve, 0, reserve);
    else
      return output(0, 0, reserve, reserve);
  }();
  auto projection = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", N, D, C,
      tensor::unbind(attention_norm_pattern),
      tensor::unbind(packed_b(q_projection, 0, D, C)),
      tensor::unbind(q_values(0, reserve, reserve)));
  auto prepared_output = matmul_op.template prepare<Parallelism>(
      workspace, "output_scratch", N, C, H * D,
      tensor::unbind(
          make_tensor(weighted_values(0, reserve, reserve, reserve).data(),
                      make_shape(N, H * D))),
      tensor::unbind(output_projection_b),
      tensor::unbind(attention_residual_pattern),
      tensor::unbind(attention_output_pattern));
  workspace.parallel_lanes<Parallelism>(
      [&](execution::TaskContext<Parallelism> task_context) {
        auto q = task_context.local(q_values);
        auto k = task_context.local(k_values);
        auto v = task_context.local(v_values);
        auto gate = task_context.local(gate_values);
        auto weighted = task_context.local(weighted_values);
        auto weighted_flat = make_tensor(weighted.data(), make_shape(N, H * D));
        auto run_task = [&]<typename Norm, typename Residual, typename Out>(
                            nint_t batch, nint_t outer, const Norm &active_norm,
                            const Residual &residual,
                            const Out
                                &active_output) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          auto batch_key_mask = key_padding_mask(batch, reserve);
          auto outer_query_mask =
              make_tensor(key_padding_mask.data() +
                              batch * key_padding_mask.stride(0) + outer,
                          make_layout(make_shape(N), make_strides(cint<0>)));
          for (nint_t head = 0; head < heads; ++head) {
            projection(task_context, active_norm,
                       packed_b(q_projection, head, D, C), q);
            projection(task_context, active_norm,
                       packed_b(k_projection, head, D, C), k);
            projection(task_context, active_norm,
                       packed_b(v_projection, head, D, C), v);
            projection(task_context, active_norm,
                       packed_b(g_projection, head, D, C), gate);

            auto active_bias = bias(batch, head, reserve, reserve);
            auto active_index = index_map(batch, outer, head, reserve, reserve);
            auto active_weight =
                weight_map(batch, outer, head, reserve, reserve);
            if (generate_maps) {
              prepared_indexer(task_context, q, k, outer_query_mask,
                               batch_key_mask, tensor::nullopt, active_bias,
                               active_index, active_weight, attention_scale);
            }

            std::mt19937 rng(static_cast<std::uint32_t>(
                seed ^ (batch * sequence * heads + outer * heads + head)));
            prepared_attention(
                task_context, q, k, v, active_index, active_weight,
                outer_query_mask, batch_key_mask, tensor::nullopt, active_bias,
                tensor::output<ComputeType>(weighted(reserve, head, reserve),
                                            sigmoid_gate_transform(gate)),
                attention_scale, rng);
          }
          prepared_output(task_context, weighted_flat, output_projection_b,
                          residual, active_output);
        };

        for (nint_t task = task_context.lane_id(); task < attention_tasks;
             task += task_context.lane_count()) {
          const nint_t batch = task / sequence;
          const nint_t outer = task % sequence;
          if constexpr (Transpose) {
            run_task(batch, outer, normalized(batch, reserve, outer, reserve),
                     act(batch, reserve, outer, reserve),
                     output(batch, reserve, outer, reserve));
          } else {
            run_task(batch, outer, normalized(batch, outer, reserve, reserve),
                     act(batch, outer, reserve, reserve),
                     output(batch, outer, reserve, reserve));
          }
        }
      });
}
