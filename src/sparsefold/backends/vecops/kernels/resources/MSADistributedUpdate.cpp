// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/** @file MSADistributedUpdate.cpp
 *  @brief Post-gather phase for distributed MSA attention.
 */

#include <algorithm>
#include <type_traits>

#include "vecops/Kernel.h"
#include "StableMath.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

using LocalExtent = spec::L;
constexpr auto M = spec::M;
constexpr auto N = spec::N;
constexpr auto C = spec::C;
constexpr auto H = spec::H;
constexpr auto D = spec::HeadDim;
using ComputeType = spec::ComputeType;

#if defined(ARCH_X86_FAMILY)
using MatmulAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using MatmulAtom = matmul::SME_BF16F32;
#else
#error "Distributed MSA update requires AMX or SME"
#endif

namespace {
template <TensorLike Packed, meta::ValueType Spatial, meta::ValueType Reduction>
auto packed_b(Packed packed, Spatial spatial, Reduction reduction) {
  const auto logical_layout = make_layout(make_shape(spatial, reduction));
  const auto layout =
      matmul::packed_layout<MatmulAtom, matmul::Operand::B>(logical_layout);
  VECOPS_ASSERT(packed.size(1) >= numel(layout), "packed B is too small");
  return make_tensor(packed.data(), layout);
}
} // namespace

void __kernel__(execution::WorkspaceContext &workspace, TensorLike auto weights,
                TensorLike auto global_value, TensorLike auto gate,
                TensorLike auto output_projection, TensorLike auto output) {
  assert_shape(weights, {vecops::any, H, vecops::any, N},
               {vecops::any, vecops::any, N, cint<1>});
  assert_shape(global_value, {N, vecops::any, M, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(gate, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});
  assert_shape(output_projection, {cint<1>, vecops::any},
               {vecops::any, cint<1>});
  assert_shape(output, {vecops::any, M, vecops::any, C},
               {vecops::any, vecops::any, C, cint<1>});

  const nint_t batch_count = output.size(0);
  const LocalExtent local_extent{output.size(2)};
  const nint_t local_rows = static_cast<nint_t>(local_extent);
  const nint_t msa_rows = static_cast<nint_t>(M);
  const nint_t heads = static_cast<nint_t>(H);
  if (batch_count == 0 || local_rows == 0)
    return;
  auto matmul_op = ops::matmul(ops::MatmulConfig<MatmulAtom>{});
  auto output_b = packed_b(output_projection, C, C);
  const nint_t tasks = batch_count * msa_rows;
  const auto weighted_shape = make_shape(LocalExtent{local_rows}, C);
  const auto weighted_head_shape = make_shape(LocalExtent{local_rows}, D);
  constexpr nint_t Parallelism =
      std::remove_cvref_t<decltype(spec::Parallelism)>::value;
  auto phase = workspace.serial_scope("msa_distributed_update");
  auto weighted_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("weighted",
                                                               weighted_shape);
  auto weighted_head_storage =
      workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "weighted_head", weighted_head_shape);

  auto prepared_head = matmul_op.template prepare<Parallelism>(
      workspace, "head_contraction_scratch", LocalExtent{local_rows}, D, N,
      tensor::unbind(weights(0, 0, reserve, reserve)),
      tensor::unbind(transpose<0, 1>(global_value(reserve, 0, 0, range(0, D)))),
      tensor::unbind(weighted_head_storage(0, reserve, reserve)));
  auto prepared_output = matmul_op.template prepare<Parallelism>(
      workspace, "output_projection_scratch", LocalExtent{local_rows}, C, C,
      tensor::unbind(weighted_storage(0, reserve, reserve)),
      tensor::unbind(output_b), tensor::unbind(output(0, 0, reserve, reserve)));

  workspace.parallel_for<Parallelism>(
      nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<Parallelism> task_context,
          const auto &work_item) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t task = work_item.begin();
        const nint_t batch = task / msa_rows;
        const nint_t msa = task % msa_rows;
        auto weighted = task_context.local(weighted_storage);
        auto weighted_head = task_context.local(weighted_head_storage);
        for (nint_t head = 0; head < heads; ++head) {
          prepared_head(task_context, weights(batch, head, reserve, reserve),
                        transpose<0, 1>(global_value(
                            reserve, batch, msa,
                            range(head * static_cast<nint_t>(D),
                                  (head + 1) * static_cast<nint_t>(D)))),
                        weighted_head);
          for (nint_t row = 0; row < local_rows; ++row)
            for (nint_t channel = 0; channel < static_cast<nint_t>(D);
                 ++channel)
              weighted(row, head * static_cast<nint_t>(D) + channel) =
                  weighted_head(row, channel);
        }
        using StorageTag = vec::ScalableTag<vecops::bfloat16_t, 0>;
        using ValueTag = vec::Rebind<ComputeType, StorageTag>;
        const nint_t lanes = vec::size(ValueTag{});
        for (nint_t row = 0; row < local_rows; ++row) {
          for (nint_t channel = 0; channel < static_cast<nint_t>(C);
               channel += lanes) {
            const nint_t active =
                std::min(lanes, static_cast<nint_t>(C) - channel);
            auto value = vec::load_convert(
                ValueTag{},
                weighted.data() + row * weighted.stride(0) + channel,
                vec::opt::first(active), vec::opt::zero);
            auto gate_value = vec::load_convert(
                ValueTag{},
                gate.data() +
                    ((batch * msa_rows + msa) * local_rows + row) *
                        static_cast<nint_t>(C) +
                    channel,
                vec::opt::first(active), vec::opt::zero);
            value = vec::mul(
                ValueTag{}, value,
                sparsefold_vecops::stable_sigmoid<ComputeType>(ValueTag{}, gate_value));
            vec::store_convert(ValueTag{},
                               weighted.data() + row * weighted.stride(0) +
                                   channel,
                               value, vec::opt::first(active));
          }
        }
        prepared_output(task_context, weighted, output_b,
                        output(batch, msa, reserve, reserve));
      });
  phase.close();
}
