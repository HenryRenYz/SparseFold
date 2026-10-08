# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import torch
from typing import Callable, Final

from sparsefold.runtime import tree
from sparsefold.runtime import util
from sparsefold.runtime import trace
from sparsefold.runtime import compiler
from sparsefold.runtime import config


def indexed_stack_scan(
    combine_fn: Callable[
                  [torch.LongTensor, tree.PyTree, tree.PyTree], tree.PyTree
                ] | Callable[
                  [torch.LongTensor, tree.PyTree], tree.PyTree
                ],
    init: tree.PyTree,
    xs: tree.PyTree | None = None,
    *,
    dim: int = 0,
    reverse: bool = False,
    length: int | None = None,
    unroll: int | bool | None = None,
    use_hop: bool = False,
) -> tree.PyTree:
  if xs is None:
    if length is None:
      raise ValueError('Either xs or length is required.')
    iter_len = length
  elif length == None:
    iter_len = len(xs)
  else:
    iter_len = length
  declared_iter_len = iter_len

  try:
    if not config.uses_vecops():
      raise ImportError
    from vecops.graph import declare_loop

    loop_trace = declare_loop(
        getattr(combine_fn, "__qualname__", "indexed_stack_scan"),
        declared_iter_len,
        attributes={
            "adapter": "sparsefold.runtime.ops.indexed_stack_scan",
            "reverse": reverse,
            "with_inputs": xs is not None,
        },
    )
  except ImportError:
    loop_trace = None

  # during precompiling, all repetitive structure only loops once
  if compiler.is_precompiling():
    # For some modules that contains caching, second pass might be different than the first
    iter_len = min(iter_len, 2)

  # Determine limits
  if reverse:
    step = -1
  else:
    step = 1

  if not use_hop:
    unroll = iter_len

  if unroll is True:
    if reverse:
      unroll_start = iter_len - 1
      unroll_end = -1
      while_end = -1
    else:
      unroll_start = 0
      unroll_end = iter_len
      while_end = iter_len
  elif isinstance(unroll, int):
    if reverse:
      unroll_start = iter_len - 1
      unroll_end = unroll_start - unroll
      while_end = -1
    else:
      unroll_start = 0
      unroll_end = unroll
      while_end = iter_len
  else:
    if reverse:
      unroll_start = iter_len - 1
      unroll_end = unroll_start
      while_end = -1
    else:
      unroll_start = 0
      unroll_end = 0
      while_end = iter_len

  is_compiling = torch.compiler.is_compiling()

  def unit(x):
    if is_compiling:
      return torch.scalar_tensor(x, dtype=torch.long, device=util.DEFAULT_DEVICE)[None]
    else:
      return x

  if xs is None:
    def body_fn(
        i: torch.LongTensor,
        carry: tree.PyTree
    ):
      carry = combine_fn(i, carry)
      return i + step, carry

    def body_fn_py(
        i: int,
        carry: tree.PyTree
    ):
      ii = (
          i
          if compiler.is_precompiling()
          else torch.scalar_tensor(
              i, dtype=torch.long, device=util.DEFAULT_DEVICE
          )
      )
      carry = combine_fn(ii, carry)
      return carry

  else:
    def body_fn(
        i: torch.LongTensor,
        carry: tree.PyTree
    ):
      # torch._constrain_as_size(i, min=0, max=iter_len)
      part = tree.map(lambda x: torch.index_select(x, dim=dim, index=i.to(device=x.device)).squeeze(dim=dim), xs)
      carry = combine_fn(i, carry, part)
      return i + step, carry

    def body_fn_py(
        i: int,
        carry: tree.PyTree
    ):
      part = tree.map(lambda x: torch.select(x, dim=dim, index=i), xs)
      ii = unit(i)
      carry = combine_fn(ii, carry, part)
      return carry

  def cond_fn_fw(
      i: torch.LongTensor,
      carry: tree.PyTree
  ):
    return i.sum() < while_end

  def cond_fn_re(
      i: torch.LongTensor,
      carry: tree.PyTree
  ):
    return i.sum() > while_end

  cond_fn = cond_fn_re if reverse else cond_fn_fw
  
  if not use_hop:
    # Only trace the function when HOP is disabled
    def body_fn_py_with_trace(i, carry):
      role = "prologue" if i == unroll_start else "steady_sample"
      if loop_trace is None:
        with trace.into(i):
          return body_fn_py(i, carry)
      with loop_trace.iteration(i, role=role, inputs=carry) as loop_state:
        with trace.into(i):
          result = body_fn_py(i, carry)
        loop_state["outputs"] = result
        return result
  else:
    body_fn_py_with_trace = body_fn_py

  # For compiler unrolling
  carry = init
  if unroll_start != unroll_end:
    for i in range(unroll_start, unroll_end, step):
      carry = body_fn_py_with_trace(i, carry)

  if unroll_end != while_end:
    init = (
      unit(unroll_end),
      # torch.zeros(1, dtype=torch.long),
      # torch.scalar_tensor(unroll_end, dtype=torch.long, device=util.DEFAULT_DEVICE)[None],
      carry
    )

    i, carry = torch.while_loop(cond_fn, body_fn, init)
    del i

  if loop_trace is not None:
    loop_trace.close()
  return carry


def stack_scan(
    combine_fn: Callable[
                  [tree.PyTree, tree.PyTree], tree.PyTree
                ] | Callable[
                  [tree.PyTree], tree.PyTree
                ],
    init: tree.PyTree,
    xs: tree.PyTree | None = None,
    *,
    dim: int = 0,
    reverse: bool = False,
    length: int | None = None,
    unroll: int | bool | None = None,
    use_hop: bool = False,
) -> tree.PyTree:
  r"""
  Performs an inclusive scan with a combine function.

  Args:
      combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
          or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
          The first input to ``combine_fn`` is the previous or initial scan carry
          and the second input element to ``combine_fn`` is a slice of the input along dim.
          The first output element of ``combine_fn`` is the next scan carry
          and the second output  of ``combine_fn`` represents a slice of the output.
          This function must be pure, i.e., no lifted arguments are supported at the moment
          and may not have any side effects.
      init (torch.Tensor or pytree with tensor leaves): The inital scan carry, a tensor, or nested pytree of tensors.
          The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
          of ``combine_fn``.
      xs (torch.Tensor or pytree with tensor leaves): The input tensor, or nested pytree of tensors.

  Kwargs:
      dim (int): the dimension to scan over, default 0.
      reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.

  Returns:
      final_carry (torch.Tensor or pytree with tensor leaves),
          the final carry of the scan operation with same pytree structure as init.

  Example::

      def add(x: torch.Tensor, y: torch.Tensor):
          next_carry = y = x + y
          return next_carry, y

      i0 = torch.zeros(1)
      xs = torch.arange(5)
      # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
      last_carry, cumsum = scan(add, init=i0, xs=xs)
  """

  if xs is None:
    def body_fn(i, carry):
      return combine_fn(carry)
  else:
    def body_fn(i, carry, xs):
      return combine_fn(carry, xs)

  return indexed_stack_scan(body_fn, init, xs, dim=dim, reverse=reverse, length=length, unroll=unroll, use_hop=use_hop)

def fori_loop(
    lower: int,
    upper: int,
    body_fn: Callable[[torch.LongTensor, tree.PyTree], tree.PyTree],
    init_val: tree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
    unroll: int | bool | None = None,
    use_hop: bool = False
) -> tree.PyTree:
  assert upper >= lower

  def body_f(i, carry):
      return body_fn(i + lower, carry)
  
  return indexed_stack_scan(body_f, init_val, dim=dim, reverse=reverse, length=upper - lower, unroll=unroll, use_hop=use_hop)
