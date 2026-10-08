// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file DiffusionCrossAttention.cpp
 *  @brief Fused conditional cross-attention used by the atom transformer.
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
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::B;
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
#error "DiffusionCrossAttention requires an AMX or SME BF16 matmul target"
#endif

namespace {

constexpr nint_t RowsPerTask = 32;
using ActiveRows = meta::Dynamic<1, 1, RowsPerTask>;
using ProjectionRows =
    std::conditional_t<(decltype(Q)::value % RowsPerTask) == 0 &&
                           (decltype(K)::value % RowsPerTask) == 0,
                       meta::Const<RowsPerTask>, ActiveRows>;
using QueryRows = std::conditional_t<(decltype(Q)::value % RowsPerTask) == 0,
                                     meta::Const<RowsPerTask>, ActiveRows>;
using AttentionConfig =
    ops::AttentionConfig<ops::MatmulConfig<MatmulAtom>, 32, 32,
                         ops::AttentionCausalMode::none,
                         ops::SDPAStrategy::automatic, vec::Accuracy::Estimate>;

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
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
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
            const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
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
        const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
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

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto x_q,
                TensorLike auto x_k, TensorLike auto mask_q,
                TensorLike auto mask_k, TensorLike auto pair_logits,
                TensorLike auto q_adaptive_scale,
                TensorLike auto q_adaptive_bias,
                TensorLike auto k_adaptive_scale,
                TensorLike auto k_adaptive_bias, TensorLike auto zero_gate,
                TensorLike auto q_projection, TensorLike auto q_bias,
                TensorLike auto k_projection, TensorLike auto v_projection,
                TensorLike auto gate_projection,
                TensorLike auto output_projection, TensorLike auto output) {
  assert_shape(x_q, {vecops::any, Q, C}, {vecops::any, C, cint<1>});
  assert_shape(x_k, {vecops::any, K, C}, {vecops::any, C, cint<1>});
  assert_shape(mask_q, {vecops::any, Q}, {vecops::any, cint<1>});
  assert_shape(mask_k, {vecops::any, K}, {vecops::any, cint<1>});
  assert_shape(pair_logits, {vecops::any, H, Q, K},
               {vecops::any, vecops::any, K, cint<1>});
  assert_shape(q_adaptive_scale, {vecops::any, Q, C},
               {vecops::any, C, cint<1>});
  assert_shape(q_adaptive_bias, {vecops::any, Q, C}, {vecops::any, C, cint<1>});
  assert_shape(k_adaptive_scale, {vecops::any, K, C},
               {vecops::any, C, cint<1>});
  assert_shape(k_adaptive_bias, {vecops::any, K, C}, {vecops::any, C, cint<1>});
  assert_shape(zero_gate, {vecops::any, Q, C}, {vecops::any, C, cint<1>});
  assert_shape(q_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(q_bias, {H, D}, {D, cint<1>});
  assert_shape(k_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(v_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(gate_projection, {H, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, Q, C}, {vecops::any, C, cint<1>});

  const auto batch = BatchExtent{x_q.size(0)};
  const nint_t batch_count = x_q.size(0);
  const nint_t queries = static_cast<nint_t>(Q);
  const nint_t keys = static_cast<nint_t>(K);
  const nint_t channels = static_cast<nint_t>(C);
  const nint_t heads = static_cast<nint_t>(H);
  const nint_t head_dim = static_cast<nint_t>(D);
  const nint_t q_mask_batches = mask_q.size(0);
  const nint_t k_mask_batches = mask_k.size(0);
  const nint_t pair_batches = pair_logits.size(0);
  const nint_t q_cond_batches = q_adaptive_scale.size(0);
  const nint_t k_cond_batches = k_adaptive_scale.size(0);
  VECOPS_ASSERT(heads * head_dim == channels,
                "CrossAttention requires H*D == C");
  VECOPS_ASSERT(batch_count % q_mask_batches == 0 &&
                    batch_count % k_mask_batches == 0 &&
                    batch_count % pair_batches == 0 &&
                    batch_count % q_cond_batches == 0 &&
                    batch_count % k_cond_batches == 0,
                "broadcast batch extents must divide the activation batch");
  if (batch_count == 0 || queries == 0 || keys == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  const auto q_head_shape = make_shape(batch, H, Q, D);
  const auto k_head_shape = make_shape(batch, H, K, D);
  auto kernel_workspace = workspace.serial_scope("diffusion_cross_attention");
  auto q = workspace.tensor<vecops::bfloat16_t>("q", q_head_shape);
  auto gate = workspace.tensor<vecops::bfloat16_t>("gate", q_head_shape);
  auto k = workspace.tensor<vecops::bfloat16_t>("k", k_head_shape);
  auto v = workspace.tensor<vecops::bfloat16_t>("v", k_head_shape);
  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto attention_op = ops::scaled_dot_product_attention(AttentionConfig{});
  auto output_projection_t = packed_b(output_projection, 0, C, C);

  const nint_t q_tiles = ceil_div(queries, RowsPerTask);
  const nint_t k_tiles = ceil_div(keys, RowsPerTask);

  const nint_t tiles_per_batch = q_tiles + k_tiles;
  const nint_t projection_tasks = batch_count * tiles_per_batch;
  auto projection_workspace = workspace.serial_scope("projection");
  const ProjectionRows projection_rows{
      std::min(std::max(queries, keys), RowsPerTask)};
  const ProjectionRows query_sample_rows{std::min(queries, RowsPerTask)};
  const ProjectionRows key_sample_rows{std::min(keys, RowsPerTask)};
  const auto channel_shape = make_shape(projection_rows, C);
  auto adapted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("adapted",
                                                               channel_shape);
  auto sample_adapted = adapted_values(0, reserve, reserve);

  auto query_adapted = sample_adapted(range(0, query_sample_rows), reserve);
  auto query_x = x_q(0, range(0, query_sample_rows), reserve);
  auto query_scale = q_adaptive_scale(0, range(0, query_sample_rows), reserve);
  auto query_bias_state =
      q_adaptive_bias(0, range(0, query_sample_rows), reserve);
  auto query_projection = q(0, 0, range(0, query_sample_rows), reserve);
  auto query_bias =
      make_tensor(q_bias.data(), make_layout(make_shape(query_sample_rows, D),
                                             make_strides(cint<0>, cint<1>)));

  auto key_adapted = sample_adapted(range(0, key_sample_rows), reserve);
  auto key_x = x_k(0, range(0, key_sample_rows), reserve);
  auto key_scale = k_adaptive_scale(0, range(0, key_sample_rows), reserve);
  auto key_bias_state = k_adaptive_bias(0, range(0, key_sample_rows), reserve);
  auto key_projection = k(0, 0, range(0, key_sample_rows), reserve);

  auto query_norm = layer_norm.template prepare<Parallelism>(
      workspace, "query_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(query_x)),
      tensor::unbind(tensor::output<ComputeType>(
          query_adapted,
          adaptive_norm_transform(query_scale, query_bias_state))));
  auto key_norm = layer_norm.template prepare<Parallelism>(
      workspace, "key_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(key_x)),
      tensor::unbind(tensor::output<ComputeType>(
          key_adapted, adaptive_norm_transform(key_scale, key_bias_state))));
  auto query_projection_op = matmul_op.template prepare<Parallelism>(
      workspace, "query_projection_scratch", query_sample_rows, D, C,
      tensor::unbind(query_adapted),
      tensor::unbind(packed_b(q_projection, 0, D, C)),
      tensor::unbind(query_bias), tensor::unbind(query_projection));
  auto query_gate_op = matmul_op.template prepare<Parallelism>(
      workspace, "query_gate_scratch", query_sample_rows, D, C,
      tensor::unbind(query_adapted),
      tensor::unbind(packed_b(gate_projection, 0, D, C)),
      tensor::unbind(query_projection));
  auto key_projection_op = matmul_op.template prepare<Parallelism>(
      workspace, "key_projection_scratch", key_sample_rows, D, C,
      tensor::unbind(key_adapted),
      tensor::unbind(packed_b(k_projection, 0, D, C)),
      tensor::unbind(key_projection));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, projection_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch_index = task / tiles_per_batch;
        const nint_t local_task = task % tiles_per_batch;
        const bool is_query = local_task < q_tiles;
        const nint_t tile = is_query ? local_task : local_task - q_tiles;
        const nint_t length = is_query ? queries : keys;
        const nint_t begin = tile * RowsPerTask;
        const nint_t end = std::min(begin + RowsPerTask, length);
        const ProjectionRows rows{end - begin};
        auto active_x = is_query ? x_q(batch_index, range(begin, end), reserve)
                                 : x_k(batch_index, range(begin, end), reserve);
        auto active_scale = is_query
                                ? q_adaptive_scale(batch_index % q_cond_batches,
                                                   range(begin, end), reserve)
                                : k_adaptive_scale(batch_index % k_cond_batches,
                                                   range(begin, end), reserve);
        auto active_bias = is_query
                               ? q_adaptive_bias(batch_index % q_cond_batches,
                                                 range(begin, end), reserve)
                               : k_adaptive_bias(batch_index % k_cond_batches,
                                                 range(begin, end), reserve);
        auto active_adapted =
            task_context.local(adapted_values)(range(0, rows), reserve);
        auto active_norm_output = tensor::output<ComputeType>(
            active_adapted, adaptive_norm_transform(active_scale, active_bias));
        if (is_query)
          query_norm(task_context, active_x, active_norm_output);
        else
          key_norm(task_context, active_x, active_norm_output);

        for (nint_t head = 0; head < heads; ++head) {
          if (is_query) {
            auto active_q_bias =
                make_tensor(q_bias.data() + head * q_bias.stride(0),
                            make_layout(make_shape(rows, D),
                                        make_strides(cint<0>, cint<1>)));
            query_projection_op(
                task_context, rows, D, C, active_adapted,
                packed_b(q_projection, head, D, C), active_q_bias,
                q(batch_index, head, range(begin, end), reserve));
            query_gate_op(task_context, rows, D, C, active_adapted,
                          packed_b(gate_projection, head, D, C),
                          gate(batch_index, head, range(begin, end), reserve));
          } else {
            key_projection_op(task_context, rows, D, C, active_adapted,
                              packed_b(k_projection, head, D, C),
                              k(batch_index, head, range(begin, end), reserve));
            key_projection_op(task_context, rows, D, C, active_adapted,
                              packed_b(v_projection, head, D, C),
                              v(batch_index, head, range(begin, end), reserve));
          }
        }
      });
  projection_workspace.close();

  const nint_t update_tasks = batch_count * q_tiles;
  auto update_workspace = workspace.serial_scope("update");
  const QueryRows update_rows{std::min(queries, RowsPerTask)};
  const auto weighted_shape = make_shape(update_rows, H, D);
  auto weighted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("weighted",
                                                               weighted_shape);
  auto sample_weighted = weighted_values(0, reserve, reserve, reserve);
  auto sample_zero_gate = zero_gate(0, range(0, update_rows), reserve);
  auto sample_weighted_flat =
      make_tensor(sample_weighted.data(), make_shape(update_rows, C));
  auto sample_q = q(0, 0, range(0, update_rows), reserve);
  auto sample_k = k(0, 0, reserve, reserve);
  auto sample_v = v(0, 0, reserve, reserve);
  auto sample_gate = gate(0, 0, range(0, update_rows), reserve);
  auto sample_pair = pair_logits(0, 0, range(0, update_rows), reserve);
  auto sample_residual = x_q(0, range(0, update_rows), reserve);
  auto sample_output = output(0, range(0, update_rows), reserve);
  const ComputeType attention_scale =
      static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  auto prepared_attention = attention_op.template prepare<Parallelism>(
      workspace, "attention_scratch", tensor::unbind(sample_q),
      tensor::unbind(sample_k), tensor::unbind(sample_v),
      tensor::unbind(mask_q(0, range(0, update_rows))),
      tensor::unbind(mask_k(0, reserve)), tensor::nullopt,
      tensor::unbind(sample_pair),
      tensor::unbind(
          tensor::output<ComputeType>(sample_weighted(reserve, 0, reserve),
                                      sigmoid_gate_transform(sample_gate))));
  auto prepared_output = matmul_op.template prepare<Parallelism>(
      workspace, "output_scratch", update_rows, C, C,
      tensor::unbind(sample_weighted_flat), tensor::unbind(output_projection_t),
      tensor::unbind(tensor::output<ComputeType>(
          sample_output,
          sigmoid_gate_residual_transform(sample_zero_gate, sample_residual))));

  workspace.parallel_lanes<Parallelism>([&](execution::TaskContext<Parallelism>
                                                task_context) {
    auto weighted = task_context.local(weighted_values);
    auto weighted_flat =
        make_tensor(weighted.data(), make_shape(update_rows, C));
    for (nint_t task = task_context.lane_id(); task < update_tasks;
         task += task_context.lane_count()) {
      const nint_t batch_index = task / q_tiles;
      const nint_t tile = task % q_tiles;
      const nint_t begin = tile * RowsPerTask;
      const nint_t end = std::min(begin + RowsPerTask, queries);
      const QueryRows rows{end - begin};
      auto active_weighted = weighted(range(0, rows), reserve, reserve);
      auto active_weighted_flat =
          make_tensor(active_weighted.data(), make_shape(rows, C));
      auto active_zero_gate =
          zero_gate(batch_index % q_cond_batches, range(begin, end), reserve);
      auto active_residual = x_q(batch_index, range(begin, end), reserve);
      auto active_output = output(batch_index, range(begin, end), reserve);
      for (nint_t head = 0; head < heads; ++head) {
        auto active_gate = gate(batch_index, head, range(begin, end), reserve);
        auto active_head_output = active_weighted(reserve, head, reserve);
        auto batch_values = v(batch_index, head, reserve, reserve);
        prepared_attention(
            task_context, q(batch_index, head, range(begin, end), reserve),
            k(batch_index, head, reserve, reserve),
            v(batch_index, head, reserve, reserve),
            mask_q(batch_index % q_mask_batches, range(begin, end)),
            mask_k(batch_index % k_mask_batches, reserve), tensor::nullopt,
            pair_logits(batch_index % pair_batches, head, range(begin, end),
                        reserve),
            tensor::output<ComputeType>(active_head_output,
                                        sigmoid_gate_transform(active_gate)),
            attention_scale);

        for (nint_t row = 0; row < rows; ++row) {
          if (mask_q.data()[(batch_index % q_mask_batches) * mask_q.stride(0) +
                            begin + row])
            continue;
          using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
          using ValueTag = vec::Rebind<ComputeType, StorageTag>;
          constexpr ValueTag value_tag{};
          const nint_t lanes = vec::size(value_tag);
          const auto inverse_keys =
              vec::fill(value_tag, static_cast<ComputeType>(1) /
                                       static_cast<ComputeType>(keys));
          for (nint_t column = 0; column < head_dim; column += lanes) {
            const nint_t active = std::min(lanes, head_dim - column);
            auto sum = vec::zeros(value_tag);
            for (nint_t key = 0; key < keys; ++key) {
              const auto values = vec::load_convert(
                  value_tag,
                  batch_values.data() + key * batch_values.stride(0) + column,
                  vec::opt::first(active), vec::opt::zero);
              sum = vec::add(value_tag, sum, values);
            }
            const auto gate_value = vec::load_convert(
                value_tag,
                active_gate.data() + row * active_gate.stride(0) + column,
                vec::opt::first(active), vec::opt::zero);
            const auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
                value_tag, gate_value);
            vec::store_convert(
                value_tag,
                active_head_output.data() + row * active_head_output.stride(0) +
                    column,
                vec::mul(value_tag, sum,
                         vec::mul(value_tag, sigmoid, inverse_keys)),
                vec::opt::first(active));
          }
        }
      }
      prepared_output(
          task_context, rows, C, C, active_weighted_flat, output_projection_t,
          tensor::output<ComputeType>(
              active_output, sigmoid_gate_residual_transform(active_zero_gate,
                                                             active_residual)));
    }
  });
}
