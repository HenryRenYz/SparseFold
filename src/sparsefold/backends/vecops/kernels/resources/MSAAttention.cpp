// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file MSAAttention.cpp @brief Fused AlphaFold 3 MSA attention. */

#include <algorithm>
#include <type_traits>
#include <utility>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/ops/Softmax.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using BatchExtent = spec::Batch;
constexpr auto M = spec::M;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto Q = spec::Q;
constexpr auto H = spec::H;
constexpr auto D = spec::HeadDim;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "MSAAttention requires an AMX or SME BF16 matmul target"
#endif

namespace {

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
                TensorLike auto mask, TensorLike auto pair_act,
                TensorLike auto act_norm_scale, TensorLike auto act_norm_offset,
                TensorLike auto pair_norm_scale,
                TensorLike auto pair_norm_offset,
                TensorLike auto pair_logits_projection,
                TensorLike auto value_projection,
                TensorLike auto gate_projection,
                TensorLike auto output_projection, TensorLike auto output) {
  static_assert(decltype(H * D)::value == decltype(C)::value,
                "MSA attention requires H * HeadDim == C");
  assert_shape(act, {vecops::any, M, N, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(mask, {vecops::any, M, N}, {vecops::any, N, cint<1>});
  assert_shape(pair_act, {vecops::any, N, N, Q},
               {vecops::any, vecops::any, Q, cint<1>});
  assert_shape(act_norm_scale, {C}, {cint<1>});
  assert_shape(act_norm_offset, {C}, {cint<1>});
  assert_shape(pair_norm_scale, {Q}, {cint<1>});
  assert_shape(pair_norm_offset, {Q}, {cint<1>});
  assert_shape(pair_logits_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(value_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(gate_projection, {cint<1>, vecops::any}, {vecops::any, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, M, N, C},
               {vecops::any, vecops::any, C, cint<1>});

  const auto batch_extent = BatchExtent{act.size(0)};
  const nint_t batch_count = act.size(0);
  const nint_t msa_rows = static_cast<nint_t>(M);
  const nint_t sequence = static_cast<nint_t>(N);
  const nint_t heads = static_cast<nint_t>(H);
  if (batch_count == 0 || msa_rows == 0 || sequence == 0)
    return;
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;

  auto layer_norm = ops::layer_norm(
      ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(1e-5)});
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto softmax_op = ops::softmax(
      ops::SoftmaxConfig<ComputeType, vec::Accuracy::Estimate>{});
  auto packer =
      ops::matmul_pack(ops::MatmulPackConfig<MatmulAtom, matmul::Operand::A>{});

  auto pair_projection_b = packed_b(pair_logits_projection, 0, H, Q);
  auto value_projection_b = packed_b(value_projection, 0, C, C);
  auto gate_projection_b = packed_b(gate_projection, 0, C, C);
  auto output_projection_b = packed_b(output_projection, 0, C, C);

  const auto key_shape = make_shape(batch_extent, N);
  const auto weight_shape = make_shape(batch_extent, H, N, N);
  const auto logical_weight_layout = make_layout(make_shape(N, N));
  const auto packed_weight_layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::A>(
          logical_weight_layout);
  const auto packed_weight_elements = numel(packed_weight_layout);
  const auto packed_weight_shape =
      make_shape(batch_extent, H, packed_weight_elements);

  auto global_scope = workspace.serial_scope("pair_weights");
  auto key_valid = workspace.tensor<int8_t>("key_valid", key_shape);
  auto weights = workspace.tensor<vecops::bfloat16_t>("weights", weight_shape);
  auto packed_weights = workspace.tensor<vecops::bfloat16_t>(
      "packed_weights", packed_weight_shape);

  workspace.parallel_for<Parallelism>(
      nint_t{0}, batch_count * sequence, cint<1>,
      [&](execution::TaskContext<Parallelism>, const auto &work_item) {
        const nint_t task = work_item.begin();
        const nint_t batch = task / sequence;
        const nint_t key = task % sequence;
        bool valid = false;
        for (nint_t row = 0; row < msa_rows; ++row)
          valid = valid || static_cast<bool>(mask(batch, row, key));
        key_valid(batch, key) = static_cast<int8_t>(valid);
      });

  auto pair_phase = workspace.serial_scope("pair_logits");
  const nint_t pair_tasks = batch_count * sequence;
  const auto normalized_pair_shape = make_shape(N, Q);
  const auto pair_logits_shape = make_shape(N, H);
  auto normalized_pairs =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "normalized_pair", normalized_pair_shape);
  auto pair_logits_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "pair_logits", pair_logits_shape);
  auto pair_norm = layer_norm.template prepare<Parallelism>(
      workspace, "pair_norm_scratch",
      tensor::unbind(
          tensor::input<ComputeType>(pair_act(0, 0, reserve, reserve))),
      tensor::unbind(tensor::input<ComputeType>(pair_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(pair_norm_offset)),
      tensor::unbind(
          tensor::output<ComputeType>(normalized_pairs(0, reserve, reserve))));
  auto pair_projection = matmul_op.template prepare<Parallelism>(
      workspace, "pair_projection_scratch", N, H, Q,
      tensor::unbind(normalized_pairs(0, reserve, reserve)),
      tensor::unbind(pair_projection_b),
      tensor::unbind(pair_logits_values(0, reserve, reserve)));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, pair_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / sequence;
        const nint_t query = task % sequence;
        auto input_pair = pair_act(batch, query, reserve, reserve);
        auto weight = weights(batch, reserve, query, reserve);
        auto normalized_pair = task_context.local(normalized_pairs);
        auto pair_logits = task_context.local(pair_logits_values);
        pair_norm(task_context, input_pair, pair_norm_scale, pair_norm_offset,
                  normalized_pair);
        pair_projection(task_context, normalized_pair, pair_projection_b,
                        pair_logits);
        for (nint_t head = 0; head < heads; ++head) {
          for (nint_t key = 0; key < sequence; ++key) {
            const ComputeType bias = key_valid(batch, key)
                                         ? static_cast<ComputeType>(0)
                                         : static_cast<ComputeType>(-1e9);
            weight(head, key) = vecops::bfloat16_t(
                static_cast<ComputeType>(pair_logits(key, head)) + bias);
          }
        }
      });
  pair_phase.close();

  auto weight_phase = workspace.serial_scope("softmax_pack");
  const nint_t weight_tasks = batch_count * heads;
  auto sample_weight = weights(0, 0, reserve, reserve);
  auto sample_packed_weight =
      make_tensor(packed_weights.data(), packed_weight_layout);
  auto prepared_softmax = softmax_op.template prepare<Parallelism>(
      workspace, "softmax_scratch",
      tensor::unbind(tensor::input<ComputeType>(sample_weight)),
      tensor::unbind(tensor::output<ComputeType>(sample_weight)));
  auto prepared_pack = packer.template prepare<Parallelism>(
      workspace, "pack_scratch", tensor::unbind(sample_weight),
      tensor::unbind(sample_packed_weight));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, weight_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / heads;
        const nint_t head = task % heads;
        auto weight = weights(batch, head, reserve, reserve);
        auto packed_weight =
            make_tensor(packed_weights.data() +
                            (batch * heads + head) *
                                static_cast<nint_t>(packed_weight_elements),
                        packed_weight_layout);
        prepared_softmax(task_context, weight, weight);
        prepared_pack(task_context, weight, packed_weight);
      });
  weight_phase.close();

  auto attention_phase = workspace.serial_scope("attention");
  const nint_t attention_tasks = batch_count * msa_rows;
  const auto activation_shape = make_shape(N, C);
  auto normalized_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "normalized", activation_shape);
  auto values = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "value", activation_shape);
  auto gates = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
      "gate", activation_shape);
  auto weighted_values =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "weighted", activation_shape);

  auto act_norm = layer_norm.template prepare<Parallelism>(
      workspace, "act_norm_scratch",
      tensor::unbind(tensor::input<ComputeType>(act(0, 0, reserve, reserve))),
      tensor::unbind(tensor::input<ComputeType>(act_norm_scale)),
      tensor::unbind(tensor::input<ComputeType>(act_norm_offset)),
      tensor::unbind(
          tensor::output<ComputeType>(normalized_values(0, reserve, reserve))));
  auto value_or_gate_projection = matmul_op.template prepare<Parallelism>(
      workspace, "value_gate_projection_scratch", N, C, C,
      tensor::unbind(normalized_values(0, reserve, reserve)),
      tensor::unbind(value_projection_b),
      tensor::unbind(values(0, reserve, reserve)));
  auto weighted_head_projection = matmul_op.template prepare<Parallelism>(
      workspace, "weighted_head_scratch", N, D, N,
      tensor::unbind(sample_packed_weight),
      tensor::unbind(transpose<0, 1>(values(0, reserve, range(0, D)))),
      tensor::unbind(weighted_values(0, reserve, range(0, D))));
  auto prepared_output_projection = matmul_op.template prepare<Parallelism>(
      workspace, "output_projection_scratch", N, C, C,
      tensor::unbind(weighted_values(0, reserve, reserve)),
      tensor::unbind(output_projection_b),
      tensor::unbind(output(0, 0, reserve, reserve)));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, attention_tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / msa_rows;
        const nint_t row = task % msa_rows;
        auto input_act = act(batch, row, reserve, reserve);
        auto output_act = output(batch, row, reserve, reserve);
        auto normalized = task_context.local(normalized_values);
        auto value = task_context.local(values);
        auto gate = task_context.local(gates);
        auto weighted = task_context.local(weighted_values);

        act_norm(task_context, input_act, act_norm_scale, act_norm_offset,
                 normalized);
        value_or_gate_projection(task_context, normalized, value_projection_b,
                                 value);
        value_or_gate_projection(task_context, normalized, gate_projection_b,
                                 gate);
        for (nint_t head = 0; head < heads; ++head) {
          auto value_head = make_tensor(
              value.data() + head * static_cast<nint_t>(D),
              make_layout(make_shape(N, D), make_strides(C, cint<1>)));
          auto weighted_head = make_tensor(
              weighted.data() + head * static_cast<nint_t>(D),
              make_layout(make_shape(N, D), make_strides(C, cint<1>)));
          auto packed_weight =
              make_tensor(packed_weights.data() +
                              (batch * heads + head) *
                                  static_cast<nint_t>(packed_weight_elements),
                          packed_weight_layout);
          weighted_head_projection(task_context, packed_weight,
                                   transpose<0, 1>(value_head), weighted_head);
        }

        using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
        using ValueTag = vec::Rebind<ComputeType, StorageTag>;
        const nint_t lanes = vec::size(ValueTag{});
        for (nint_t query = 0; query < sequence; ++query) {
          for (nint_t channel = 0; channel < static_cast<nint_t>(C);
               channel += lanes) {
            const nint_t active =
                std::min(lanes, static_cast<nint_t>(C) - channel);
            auto value_vector = vec::load_convert(
                ValueTag{},
                weighted.data() + query * weighted.stride(0) + channel,
                vec::opt::first(active), vec::opt::zero);
            auto gate_vector = vec::load_convert(
                ValueTag{}, gate.data() + query * gate.stride(0) + channel,
                vec::opt::first(active), vec::opt::zero);
            auto sigmoid = sparsefold_vecops::stable_sigmoid<ComputeType>(
                ValueTag{}, gate_vector);
            vec::store_convert(ValueTag{},
                               weighted.data() + query * weighted.stride(0) +
                                   channel,
                               vec::mul(ValueTag{}, value_vector, sigmoid),
                               vec::opt::first(active));
          }
        }
        prepared_output_projection(task_context, weighted, output_projection_b,
                                   output_act);
      });
  attention_phase.close();
  global_scope.close();
}
