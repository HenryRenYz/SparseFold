// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file DiffusionCrossAttentionIndexed.cpp
 *  @brief Atom cross-attention that projects unique atoms before window gather.
 */

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Attention.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::B;
constexpr auto A = spec::A;
constexpr auto U = spec::U;
constexpr auto Q = spec::Q;
constexpr auto K = spec::K;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::D;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "DiffusionCrossAttentionIndexed requires an AMX or SME BF16 target"
#endif

namespace {

constexpr nint_t RowsPerTask = 32;
using ActiveRows = meta::Dynamic<1, 1, RowsPerTask>;
using ProjectionRows =
    std::conditional_t<(decltype(A)::value % RowsPerTask) == 0,
                       meta::Const<RowsPerTask>, ActiveRows>;

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
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<
            ComputeType, vec::Accuracy::Estimate>(
            tag, scale_value);
        return vec::add(tag, vec::mul(tag, sigmoid, value), bias_value);
      });
}

template <TensorLike Gate> auto sigmoid_gate_transform(Gate gate) {
  const auto *gate_data = gate.data();
  const nint_t gate_stride = gate.stride(0);
  return make_lane_local_vec_transform<ComputeType, ComputeType>(
      [gate_data, gate_stride](auto tag, auto value, const auto &context)
          VECOPS_KERNEL_LAMBDA {
            const auto coordinate = context.lane_coord(0);
            const nint_t row = coordinate[0];
            const nint_t column = coordinate[1];
            const auto gate_value =
                vec::load_convert(tag, gate_data + row * gate_stride + column,
                                  context.active_option(tag), vec::opt::zero);
            const auto sigmoid = sparsefold_vecops::stable_sigmoid<
                ComputeType, vec::Accuracy::Estimate>(
                tag, gate_value);
            return vec::mul(tag, value, sigmoid);
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
        const auto active = context.active_option(tag);
        const auto gate_value =
            vec::load_convert(tag, gate_data + row * gate_stride + column,
                              active, vec::opt::zero);
        const auto residual_value = vec::load_convert(
            tag, residual_data + row * residual_stride + column, active,
            vec::opt::zero);
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<
            ComputeType, vec::Accuracy::Estimate>(
            tag, gate_value);
        return vec::add(tag, vec::mul(tag, value, sigmoid), residual_value);
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
                TensorLike auto query_mask, TensorLike auto key_indices,
                TensorLike auto key_mask, TensorLike auto pair_logits,
                TensorLike auto q_adaptive_scale,
                TensorLike auto q_adaptive_bias,
                TensorLike auto k_adaptive_scale,
                TensorLike auto k_adaptive_bias, TensorLike auto zero_gate,
                TensorLike auto q_projection, TensorLike auto q_bias,
                TensorLike auto k_projection, TensorLike auto v_projection,
                TensorLike auto gate_projection,
                TensorLike auto output_projection, TensorLike auto output) {
  assert_shape(act, {vecops::any, A, C}, {vecops::any, C, cint<1>});
  assert_shape(query_mask, {U, Q}, {Q, cint<1>});
  assert_shape(key_indices, {U, K}, {K, cint<1>});
  assert_shape(key_mask, {U, K}, {K, cint<1>});
  assert_shape(pair_logits, {U, H, Q, K},
               {vecops::any, vecops::any, K, cint<1>});
  assert_shape(q_adaptive_scale, {A, C}, {C, cint<1>});
  assert_shape(q_adaptive_bias, {A, C}, {C, cint<1>});
  assert_shape(k_adaptive_scale, {A, C}, {C, cint<1>});
  assert_shape(k_adaptive_bias, {A, C}, {C, cint<1>});
  assert_shape(zero_gate, {A, C}, {C, cint<1>});
  assert_shape(q_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(q_bias, {H, D}, {D, cint<1>});
  assert_shape(k_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(v_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(gate_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, A, C}, {vecops::any, C, cint<1>});
  static_assert(decltype(U * Q)::value == decltype(A)::value,
                "unique atom count must equal subsets times queries");
  static_assert(decltype(H * D)::value == decltype(C)::value,
                "CrossAttention requires H*D == C");

  const auto batch = BatchExtent{act.size(0)};
  const nint_t batch_count = act.size(0);
  const nint_t atoms = static_cast<nint_t>(A);
  const nint_t subsets = static_cast<nint_t>(U);
  const nint_t queries = static_cast<nint_t>(Q);
  const nint_t keys = static_cast<nint_t>(K);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  if (batch_count == 0 || atoms == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const auto head_shape = make_shape(batch, H, A, D);
  auto global_scope = workspace.serial_scope("projected_global");
  auto q = workspace.tensor<vecops::bfloat16_t>("q", head_shape);
  auto k = workspace.tensor<vecops::bfloat16_t>("k", head_shape);
  auto v = workspace.tensor<vecops::bfloat16_t>("v", head_shape);
  auto gate = workspace.tensor<vecops::bfloat16_t>("gate", head_shape);

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto pack_a_op =
      ops::matmul_pack(ops::MatmulPackConfig<MatmulAtom, matmul::Operand::A>{});
  auto attention_op = ops::scaled_dot_product_attention(
      ops::AttentionConfig<ops::MatmulConfig<MatmulAtom>, 32, 32,
                           ops::AttentionCausalMode::none,
                           ops::SDPAStrategy::automatic,
                           vec::Accuracy::Estimate>{});
  auto output_projection_t = packed_b(output_projection, 0, C, C);

  const nint_t projection_tiles = ceil_div(atoms, RowsPerTask);
  const ProjectionRows projection_rows{std::min(atoms, RowsPerTask)};
  // Normalize into worker-local tiles, pack Q/K once, then project every head.
  // This removes two full [B,A,C] intermediates, one team barrier, and the
  // former per-head repetition of activation packing.
  const nint_t projection_tasks = batch_count * projection_tiles;
  auto projection_scope = workspace.serial_scope("normalize_and_project");
  const auto projection_shape = make_shape(projection_rows, C);
  const auto packed_projection_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
          make_layout(projection_shape));
  auto q_adapted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "q_adapted", projection_shape);
  auto k_adapted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "k_adapted", projection_shape);
  auto packed_q_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "packed_q", packed_projection_layout);
  auto packed_k_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "packed_k", packed_projection_layout);
  auto sample_q_adapted = q_adapted_values(0, reserve, reserve);
  auto sample_k_adapted = k_adapted_values(0, reserve, reserve);
  auto sample_packed_q_adapted = packed_q_values(0, ellipsis);
  auto sample_packed_k_adapted = packed_k_values(0, ellipsis);
  auto sample_projection = q(0, 0, range(0, projection_rows), reserve);
  auto sample_q_bias =
      make_tensor(q_bias.data(), make_layout(make_shape(projection_rows, D),
                                             make_strides(cint<0>, cint<1>)));
  auto sample_act = act(0, range(0, projection_rows), reserve);
  auto q_norm = layer_norm.template prepare<Parallelism>(
      workspace, "q_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_act)),
      tensor::unbind(tensor::output<ComputeType>(
          sample_q_adapted,
          adaptive_norm_transform(
              q_adaptive_scale(range(0, projection_rows), reserve),
              q_adaptive_bias(range(0, projection_rows), reserve)))));
  auto k_norm = layer_norm.template prepare<Parallelism>(
      workspace, "k_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_act)),
      tensor::unbind(tensor::output<ComputeType>(
          sample_k_adapted,
          adaptive_norm_transform(
              k_adaptive_scale(range(0, projection_rows), reserve),
              k_adaptive_bias(range(0, projection_rows), reserve)))));
  auto pack_adapted = pack_a_op.template prepare<Parallelism>(
      workspace, "pack_scratch", tensor::unbind(sample_q_adapted),
      tensor::unbind(sample_packed_q_adapted));
  auto q_projection_op = matmul_op.template prepare<Parallelism>(
      workspace, "q_projection_scratch", projection_rows, D, C,
      tensor::unbind(sample_packed_q_adapted),
      tensor::unbind(packed_b(q_projection, 0, D, C)),
      tensor::unbind(sample_q_bias), tensor::unbind(sample_projection));
  auto kvg_projection_op = matmul_op.template prepare<Parallelism>(
      workspace, "kvg_projection_scratch", projection_rows, D, C,
      tensor::unbind(sample_packed_k_adapted),
      tensor::unbind(packed_b(k_projection, 0, D, C)),
      tensor::unbind(sample_projection));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, projection_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        auto q_adapted = task_context.local(q_adapted_values);
        auto k_adapted = task_context.local(k_adapted_values);
        auto packed_q_storage = task_context.local(packed_q_values);
        auto packed_k_storage = task_context.local(packed_k_values);
        const nint_t task = work_item.begin();
        const nint_t batch_index = task / projection_tiles;
        const nint_t tile = task % projection_tiles;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, atoms);
        const ProjectionRows rows{end - begin};
        auto active_q_adapted = q_adapted(range(0, rows), reserve);
        auto active_k_adapted = k_adapted(range(0, rows), reserve);
        auto active_act = act(batch_index, range(begin, end), reserve);
        q_norm(task_context, active_act,
               tensor::output<ComputeType>(
                   active_q_adapted,
                   adaptive_norm_transform(
                       q_adaptive_scale(range(begin, end), reserve),
                       q_adaptive_bias(range(begin, end), reserve))));
        k_norm(task_context, active_act,
               tensor::output<ComputeType>(
                   active_k_adapted,
                   adaptive_norm_transform(
                       k_adaptive_scale(range(begin, end), reserve),
                       k_adaptive_bias(range(begin, end), reserve))));
        auto active_packed_layout =
            matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
                active_q_adapted.layout());
        auto active_packed_q =
            make_tensor(packed_q_storage.data(), active_packed_layout);
        auto active_packed_k =
            make_tensor(packed_k_storage.data(), active_packed_layout);
        pack_adapted(task_context, active_q_adapted, active_packed_q);
        pack_adapted(task_context, active_k_adapted, active_packed_k);
        for (nint_t head = 0; head < heads; ++head) {
          auto active_q_bias = make_tensor(
              q_bias.data() + head * q_bias.stride(0),
              make_layout(make_shape(rows, D), make_strides(cint<0>, cint<1>)));
          q_projection_op(task_context, rows, D, C, active_packed_q,
                          packed_b(q_projection, head, D, C), active_q_bias,
                          q(batch_index, head, range(begin, end), reserve));
          kvg_projection_op(
              task_context, rows, D, C, active_packed_q,
              packed_b(gate_projection, head, D, C),
              gate(batch_index, head, range(begin, end), reserve));
          kvg_projection_op(task_context, rows, D, C, active_packed_k,
                            packed_b(k_projection, head, D, C),
                            k(batch_index, head, range(begin, end), reserve));
          kvg_projection_op(task_context, rows, D, C, active_packed_k,
                            packed_b(v_projection, head, D, C),
                            v(batch_index, head, range(begin, end), reserve));
        }
      });
  projection_scope.close();

  // One task owns all heads of a (sample, subset).  Keep weighted heads local
  // and immediately run the output projection, avoiding a global [B,A,C]
  // intermediate and a fourth OpenMP region.
  const nint_t attention_tasks = batch_count * subsets;
  auto attention_scope = workspace.serial_scope("indexed_attention_and_output");
  const auto key_shape = make_shape(K, D);
  const auto weighted_shape = make_shape(Q, H, D);
  auto key_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "keys", key_shape);
  auto value_values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "values", key_shape);
  auto weighted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("weighted",
                                                               weighted_shape);
  auto sample_keys = key_values(0, reserve, reserve);
  auto sample_values = value_values(0, reserve, reserve);
  auto sample_weighted = weighted_values(0, reserve, reserve, reserve);
  auto sample_query = q(0, 0, range(0, Q), reserve);
  auto sample_gate = gate(0, 0, range(0, Q), reserve);
  auto sample_pair = pair_logits(0, 0, reserve, reserve);
  auto sample_weighted_head = sample_weighted(reserve, 0, reserve);
  auto sample_weighted_flat =
      make_tensor(sample_weighted.data(), make_shape(Q, C));
  const ComputeType attention_scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "attention_scratch", tensor::unbind(sample_query),
      tensor::unbind(sample_keys), tensor::unbind(sample_values),
      tensor::unbind(query_mask(0, reserve)),
      tensor::unbind(key_mask(0, reserve)), tensor::nullopt,
      tensor::unbind(sample_pair),
      tensor::unbind(tensor::output<ComputeType>(
          sample_weighted_head, sigmoid_gate_transform(sample_gate))));
  auto prepared_output = matmul_op.template prepare<Parallelism>(
      workspace, "output_scratch", Q, C, C,
      tensor::unbind(sample_weighted_flat), tensor::unbind(output_projection_t),
      tensor::unbind(tensor::output<ComputeType>(
          output(0, range(0, Q), reserve),
          sigmoid_gate_residual_transform(zero_gate(range(0, Q), reserve),
                                          act(0, range(0, Q), reserve)))));
  workspace.parallel_lanes<Parallelism>([&](execution::TaskContext<Parallelism>
                                                task_context) {
    auto local_keys = task_context.local(key_values);
    auto local_values = task_context.local(value_values);
    auto local_weighted = task_context.local(weighted_values);
    for (nint_t task = task_context.lane_id(); task < attention_tasks;
         task += task_context.lane_count()) {
      const nint_t batch_index = task / subsets;
      const nint_t subset = task % subsets;
      const nint_t begin = subset * queries;
      for (nint_t head = 0; head < heads; ++head) {
        for (nint_t key_index = 0; key_index < keys; ++key_index) {
          const nint_t source = static_cast<nint_t>(
              key_indices.data()[subset * key_indices.stride(0) + key_index]);
          const bool source_valid = source >= 0 && source < atoms;
          const nint_t safe_source = source_valid ? source : 0;
          auto *key_destination =
              local_keys.data() + key_index * local_keys.stride(0);
          auto *value_destination =
              local_values.data() + key_index * local_values.stride(0);
          const auto *key_source = k.data() + batch_index * k.stride(0) +
                                   head * k.stride(1) +
                                   safe_source * k.stride(2);
          const auto *value_source = v.data() + batch_index * v.stride(0) +
                                     head * v.stride(1) +
                                     safe_source * v.stride(2);
          using CopyTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
          constexpr CopyTag copy_tag{};
          const nint_t copy_lanes = vec::size(copy_tag);
          for (nint_t column = 0; column < head_dim; column += copy_lanes) {
            const nint_t active = std::min(copy_lanes, head_dim - column);
            const auto key_values =
                source_valid
                    ? vec::load(copy_tag, key_source + column,
                                vec::opt::first(active), vec::opt::zero)
                    : vec::zeros(copy_tag);
            const auto value_values =
                source_valid
                    ? vec::load(copy_tag, value_source + column,
                                vec::opt::first(active), vec::opt::zero)
                    : vec::zeros(copy_tag);
            vec::store(copy_tag, key_destination + column, key_values,
                       vec::opt::first(active));
            vec::store(copy_tag, value_destination + column, value_values,
                       vec::opt::first(active));
          }
        }
        auto active_gate =
            gate(batch_index, head, range(begin, begin + queries), reserve);
        auto active_head_output = local_weighted(reserve, head, reserve);
        prepared_attention(
            task_context,
            q(batch_index, head, range(begin, begin + queries), reserve),
            local_keys, local_values, query_mask(subset, reserve),
            key_mask(subset, reserve), tensor::nullopt,
            pair_logits(subset, head, reserve, reserve),
            tensor::output<ComputeType>(active_head_output,
                                        sigmoid_gate_transform(active_gate)),
            attention_scale);

        for (nint_t row = 0; row < queries; ++row) {
          if (query_mask.data()[subset * query_mask.stride(0) + row])
            continue;
          // Masked atom queries never become valid K/V sources and are
          // masked again after the transformer. Avoid the generic
          // attention fallback's otherwise dominant reduction over all
          // 128 V rows in the heavily padded 6144-row layout.
          using ZeroTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
          constexpr ZeroTag zero_tag{};
          const nint_t lanes = vec::size(zero_tag);
          const auto zeros = vec::zeros(zero_tag);
          for (nint_t column = 0; column < head_dim; column += lanes) {
            const nint_t active = std::min(lanes, head_dim - column);
            vec::store(zero_tag,
                       active_head_output.data() +
                           row * active_head_output.stride(0) + column,
                       zeros, vec::opt::first(active));
          }
        }
      }
      auto active_weighted_flat =
          make_tensor(local_weighted.data(), make_shape(Q, C));
      prepared_output(
          task_context, active_weighted_flat, output_projection_t,
          tensor::output<ComputeType>(
              output(batch_index, range(begin, begin + queries), reserve),
              sigmoid_gate_residual_transform(
                  zero_gate(range(begin, begin + queries), reserve),
                  act(batch_index, range(begin, begin + queries), reserve))));
    }
  });
}
