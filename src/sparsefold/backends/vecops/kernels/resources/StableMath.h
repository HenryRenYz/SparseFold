// SPDX-FileCopyrightText: 2026 HenryRenYz
// SPDX-License-Identifier: MIT

/**
 * Vector math helpers shared by AF3 kernels.
 *
 * SME uses a polynomial sigmoid to avoid the data-dependent FEXPA latency
 * observed during late diffusion steps on 920F. Other targets retain their
 * accuracy-selected vector exponential implementation.
 */
#pragma once

#include "vecops/Kernel.h"

namespace sparsefold_vecops {

using namespace ::vecops;

// Avoid the high-latency FEXPA tail on 920F. The reduced argument stays in
// [-0.625, 0], where a degree-nine polynomial is accurate; seven squarings
// reconstruct exp(-|x|) without entering the subnormal range.
template <typename Scalar, typename Tag, typename Value>
VECOPS_INLINE auto polynomial_sigmoid(Tag tag, Value value) {
  const auto one = vec::fill(tag, static_cast<Scalar>(1));
  const auto magnitude = vec::min(
      tag, vec::abs(tag, value), vec::fill(tag, static_cast<Scalar>(80)));
  const auto x = vec::neg(tag, vec::mul(
      tag, magnitude, vec::fill(tag, static_cast<Scalar>(1.0f / 128.0f))));
  auto exp_abs = vec::fill(tag, static_cast<Scalar>(1.0f / 362880.0f));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(1.0f / 40320.0f)));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(1.0f / 5040.0f)));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(1.0f / 720.0f)));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(1.0f / 120.0f)));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(1.0f / 24.0f)));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(1.0f / 6.0f)));
  exp_abs = vec::fmadd(tag, exp_abs, x,
                       vec::fill(tag, static_cast<Scalar>(0.5f)));
  exp_abs = vec::fmadd(tag, exp_abs, x, one);
  exp_abs = vec::fmadd(tag, exp_abs, x, one);
  for (int i = 0; i < 7; ++i)
    exp_abs = vec::mul(tag, exp_abs, exp_abs);
  const auto denominator = vec::add(tag, one, exp_abs);
  const auto positive = vec::div(tag, one, denominator);
  const auto negative = vec::div(tag, exp_abs, denominator);
  return vec::blend(tag, negative,
                    vec::cmpge(tag, value, vec::zeros(tag)), positive);
}

template <typename Scalar, vec::Accuracy ExpAccuracy = vec::Accuracy::Fast,
          typename Tag, typename Value>
VECOPS_INLINE auto stable_sigmoid(Tag tag, Value value) {
#if defined(HAS_SME)
  return polynomial_sigmoid<Scalar>(tag, value);
#else
  const auto one = vec::fill(tag, static_cast<Scalar>(1));
  // exp(-87.3) reaches the FP32 subnormal range. Sigmoid is already
  // saturated to useful precision at |x| = 80.
  const auto magnitude = vec::min(
      tag, vec::abs(tag, value), vec::fill(tag, static_cast<Scalar>(80)));
  const auto exponent = vec::neg(
      tag, vec::mul(tag, magnitude, vec::fill(tag, static_cast<Scalar>(0.125))));
  auto exp_abs = [&] {
    if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
      return vec::exp_neg_est(tag, exponent);
    } else if constexpr (ExpAccuracy == vec::Accuracy::Fast) {
      return vec::exp_neg_fast(tag, exponent);
    } else {
      return vec::exp(tag, exponent);
    }
  }();
  exp_abs = vec::mul(tag, exp_abs, exp_abs);
  exp_abs = vec::mul(tag, exp_abs, exp_abs);
  exp_abs = vec::mul(tag, exp_abs, exp_abs);
  const auto denominator = vec::add(tag, one, exp_abs);
  const auto positive = vec::div(tag, one, denominator);
  const auto negative = vec::div(tag, exp_abs, denominator);
  return vec::blend(tag, negative,
                    vec::cmpge(tag, value, vec::zeros(tag)), positive);
#endif
}

}  // namespace sparsefold_vecops
