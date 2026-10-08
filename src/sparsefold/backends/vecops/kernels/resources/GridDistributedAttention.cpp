// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file GridDistributedAttention.cpp
 *  @brief Local post-collective phase for distributed IBS grid attention.
 */

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
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using LocalExtent = spec::L;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::HeadDim;
constexpr auto T = spec::T;
constexpr auto S = spec::S;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed grid attention requires AMX or SME"
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
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            const auto coordinate = context.lane_coord(0);
            const auto gate_value = vec::load_convert(
                tag, gate_data + coordinate[0] * gate_stride + coordinate[1],
                context.active_option(tag), vec::opt::zero);
            return vec::mul(
                tag, value,
                sparsefold_vecops::stable_sigmoid<ComputeType>(tag, gate_value));
          });
}

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

void __kernel__(execution::WorkspaceContext &workspace,
                TensorLike auto normalized, TensorLike auto key_padding_mask,
                TensorLike auto global_bias, TensorLike auto q_projection,
                TensorLike auto k_projection, TensorLike auto v_projection,
                TensorLike auto output_projection, TensorLike auto g_projection,
                TensorLike auto index_map, TensorLike auto weight_map,
                TensorLike auto output, int64_t row_offset,
                int64_t random_blocks, float32_t static_probability,
                float32_t random_probability, float32_t minimum_probability,
                bool generate_maps, int64_t seed) {
  assert_shape(normalized, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(key_padding_mask, {vecops::any, N}, {vecops::any, cint<1>});
  assert_shape(global_bias, {N, vecops::any, H, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(q_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(k_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(v_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(g_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(index_map, {vecops::any, vecops::any, H, T, S},
               {vecops::any, vecops::any, vecops::any, S, cint<1>});
  assert_shape(weight_map, {vecops::any, vecops::any, H, T, S},
               {vecops::any, vecops::any, vecops::any, S, cint<1>});
  assert_shape(output, {vecops::any, vecops::any, N, C},
               {vecops::any, vecops::any, C, cint<1>});

  const nint_t batch_count = normalized.size(0);
  const LocalExtent local_extent{normalized.size(1)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  if (batch_count == 0 || local_rows == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  AttentionConfig attention_config{};
  attention_config.selected_blocks = static_cast<nint_t>(S);
  attention_config.random_blocks = static_cast<nint_t>(random_blocks);
  attention_config.static_probability = static_probability;
  attention_config.random_probability = random_probability;
  attention_config.minimum_probability = minimum_probability;
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto indexer = ops::ibs_attention_indexer(attention_config);
  auto attention_op = ops::ibs_attention(attention_config);
  auto output_b = packed_b(output_projection, 0, C, H * D);
  const ComputeType attention_scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  const nint_t tasks = batch_count * local_rows;
  auto phase = workspace.serial_scope("grid_distributed_attention");
  const auto head_shape = make_shape(N, D);
  const auto weighted_shape = make_shape(N, H, D);
  auto q_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("q", head_shape);
  auto k_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("k", head_shape);
  auto v_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("v", head_shape);
  auto gate_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "gate", head_shape);
  auto weighted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("weighted",
                                                               weighted_shape);
  auto sample_q = q_values(0, reserve, reserve);
  auto sample_k = k_values(0, reserve, reserve);
  auto sample_v = v_values(0, reserve, reserve);
  auto sample_gate = gate_values(0, reserve, reserve);
  auto sample_weighted = weighted_values(0, reserve, reserve, reserve);
  auto sample_weighted_flat =
      make_tensor(sample_weighted.data(), make_shape(N, H * D));
  auto sample_query_mask =
      make_tensor(key_padding_mask.data(),
                  make_layout(make_shape(N), make_strides(cint<0>)));
  auto projection = matmul_op.template prepare<Parallelism>(
      workspace, "projection_scratch", N, D, C,
      tensor::unbind(normalized(0, 0, reserve, reserve)),
      tensor::unbind(packed_b(q_projection, 0, D, C)),
      tensor::unbind(sample_q));
  auto output_projection_op = matmul_op.template prepare<Parallelism>(
      workspace, "output_scratch", N, C, H * D,
      tensor::unbind(sample_weighted_flat), tensor::unbind(output_b),
      tensor::unbind(output(0, 0, reserve, reserve)));
  auto prepared_indexer = indexer.template prepare<Parallelism>(
      workspace, "index_scratch", tensor::unbind(sample_q),
      tensor::unbind(sample_k), tensor::unbind(sample_query_mask),
      tensor::unbind(key_padding_mask(0, reserve)), tensor::nullopt,
      tensor::unbind(global_bias(reserve, 0, 0, reserve)),
      tensor::unbind(index_map(0, 0, 0, reserve, reserve)),
      tensor::unbind(weight_map(0, 0, 0, reserve, reserve)));
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "attention_scratch", tensor::unbind(sample_q),
      tensor::unbind(sample_k), tensor::unbind(sample_v),
      tensor::unbind(index_map(0, 0, 0, reserve, reserve)),
      tensor::unbind(weight_map(0, 0, 0, reserve, reserve)),
      tensor::unbind(sample_query_mask),
      tensor::unbind(key_padding_mask(0, reserve)), tensor::nullopt,
      tensor::unbind(global_bias(reserve, 0, 0, reserve)),
      tensor::unbind(
          tensor::output<ComputeType>(sample_weighted(reserve, 0, reserve),
                                      sigmoid_gate_transform(sample_gate))));

  workspace.parallel_lanes<Parallelism>(
      [&](execution::TaskContext<Parallelism> task_context) {
        auto q = task_context.local(q_values);
        auto k = task_context.local(k_values);
        auto v = task_context.local(v_values);
        auto gate = task_context.local(gate_values);
        auto weighted = task_context.local(weighted_values);
        auto weighted_flat = make_tensor(weighted.data(), make_shape(N, H * D));
        for (nint_t task = task_context.lane_id(); task < tasks;
             task += task_context.lane_count()) {
          const nint_t batch = task / local_rows;
          const nint_t outer = task % local_rows;
          const nint_t global_outer = row_offset + outer;
          auto active_norm = normalized(batch, outer, reserve, reserve);
          auto active_query_mask =
              make_tensor(key_padding_mask.data() +
                              batch * key_padding_mask.stride(0) + global_outer,
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
            auto active_bias = global_bias(reserve, batch, head, reserve);
            auto active_index = index_map(batch, outer, head, reserve, reserve);
            auto active_weight =
                weight_map(batch, outer, head, reserve, reserve);
            if (generate_maps) {
              prepared_indexer(task_context, q, k, active_query_mask,
                               key_padding_mask(batch, reserve),
                               tensor::nullopt, active_bias, active_index,
                               active_weight, attention_scale);
            }
            std::mt19937 rng(static_cast<std::uint32_t>(
                seed ^
                (batch * sequence * heads + global_outer * heads + head)));
            prepared_attention(
                task_context, q, k, v, active_index, active_weight,
                active_query_mask, key_padding_mask(batch, reserve),
                tensor::nullopt, active_bias,
                tensor::output<ComputeType>(weighted(reserve, head, reserve),
                                            sigmoid_gate_transform(gate)),
                attention_scale, rng);
          }
          output_projection_op(task_context, weighted_flat, output_b,
                               output(batch, outer, reserve, reserve));
        }
      });
  phase.close();
}
