# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Specialized mapping functions."""

import functools
import torch
from collections.abc import Callable, Sequence
from typing import Any

from sparsefold.runtime import tree
from sparsefold.runtime import config
from sparsefold.runtime import trace
from sparsefold.runtime import compiler

Pytree = Any
PytreeTensor = Any

partial = functools.partial
PROXY = object()


def _maybe_slice(array: torch.Tensor, i, slice_size, dim):
  if dim is PROXY:
    return array
  else:
    indices = torch.arange(i, i + slice_size, device=array.device)
    return torch.index_select(array, dim=dim, index=indices)


def _maybe_get_size(array, axis):
  if axis == PROXY:
    return -1
  else:
    return array.shape[axis]


def _replace_nones(sentinel, tree_):
  """Replaces ``None`` in ``tree`` with ``sentinel``."""
  leaves, treedef = tree.flatten(tree_)
  leaves = map(lambda x: sentinel if x is None else x, leaves)
  return treedef.unflatten(leaves)


def _flatten_axes(name, treespec: tree.TreeSpec, axis_tree, *, kws=False, tupled_args=False):
  # given an axis spec tree axis_tree (a pytree with integers and Nones at the
  # leaves, i.e. the Nones are to be considered leaves) that is a tree prefix of
  # the given treespec, build a complete axis spec tree with the same structure
  # and return the flattened result
  # TODO(mattjj,phawkins): improve this implementation
  proxy = object()
  dummy = tree.unflatten([object()] * treespec.num_leaves, treespec)
  axes = []
  add_leaves = lambda i, x: axes.extend([i] * len(tree.flatten(x)[0]))
  try:
    tree.map(add_leaves, _replace_nones(proxy, axis_tree), dummy)
  except ValueError:
    if kws:
      # if keyword arguments are included in the tree, we make adapt the error
      # message only to be about the positional arguments
      treespec, _ = treespec.children_specs
      axis_tree, _ = axis_tree
    hint = ""
    if tupled_args:
      hint += (f" Note that {name} that are non-trivial pytrees should always be "
               f"wrapped in a tuple representing the argument list.")
      if treespec.num_children == 1:
        try:
          _flatten_axes(name, treespec, (axis_tree,))
        except ValueError:
          pass  # That's not the issue.
        else:
          hint += (f" In particular, you're passing in a single argument which "
                   f"means that {name} might need to be wrapped in "
                   f"a singleton tuple.")
    raise ValueError(f"{name} specification must be a tree prefix of the "
                     f"corresponding value, got specification {axis_tree} "
                     f"for value tree {treespec}.{hint}") from None
  axes = [None if a is proxy else a for a in axes]
  assert len(axes) == treespec.num_leaves
  return axes


def _expand_axes(axes, values, name="sharded_apply"):
  values_tree_def = tree.structure(values)
  flat_axes = _flatten_axes(name, values_tree_def, axes)
  # Replace None's with PROXY.
  flat_axes = [PROXY if x is None else x for x in flat_axes]
  return tree.unflatten(flat_axes, values_tree_def)


def sharded_map(
    fun: Callable[..., PytreeTensor],
    shard_size: int | None = 1,
    in_axes: int = 0,
    out_axes: int = 0,
) -> Callable[..., PytreeTensor]:
  """Sharded vmap.

  Maps `fun` over axes, in a way similar to hk.vmap, but does so in shards of
  `shard_size`. This allows a smooth trade-off between memory usage
  (as in a plain map) vs higher throughput (as in a vmap).

  Args:
    fun: Function to apply smap transform to.
    shard_size: Integer denoting shard size.
    in_axes: Either integer or pytree describing which axis to map over for each
      input to `fun`, None denotes broadcasting.
    out_axes: Integer or pytree denoting to what axis in the output the mapped
      over axis maps.

  Returns:
    Function with smap applied.
  """
  vmapped_fun = torch.vmap(fun, in_axes, out_axes)
  return sharded_apply(vmapped_fun, shard_size, in_axes, out_axes)


# ASC: Target fun wrapper 
@torch.compiler.disable
def _apply_fun_to_slice(
    in_axes: int,
    fun: Callable[..., PytreeTensor],
    args: tuple,
    kwargs: dict,
    slice_start: int,
    slice_size: int
):
  input_slice = tree.map(
    lambda array: _maybe_slice(
      array, slice_start, slice_size, in_axes
    ),
    args,
  )
  return fun(*input_slice, **kwargs)


def _merge(out_axes, *tensors):
  return torch.cat(tensors, dim=out_axes)


def _get_shape_dim(in_axes, array: torch.Tensor):
  if in_axes < 0:
    return array.shape[array.ndim + in_axes]
  else:
    return array.shape[in_axes]


def _mapped_fn(
    fun: Callable[..., PytreeTensor],
    in_axes: int,
    out_axes: int,
    shard_size: int,
    *args,
    **kwargs):
  """
  Mapped version of {fun}. Takes similar arguments to {fun} 
  but with additional array axes over which {fun} is mapped.
  """
  # Expand in axes and determine loop range.
  # in_axes_ = _expand_axes(in_axes, args)

  # in_sizes = tree_map(_maybe_get_size, args, in_axes_)
  get_shape_dim = partial(_get_shape_dim, in_axes)
  in_sizes = tree.map(get_shape_dim, args)
  in_size = max(tree.leaves(in_sizes))

  num_extra_shards = (in_size - 1) // shard_size

  # Fix if necessary.
  last_shard_size = in_size % shard_size
  last_shard_size = shard_size if last_shard_size == 0 else last_shard_size

  # A shard larger than the input has no full slices, only the remainder.
  # torch.arange rejects a negative stop with a positive step.
  full_shard_stop = max(0, in_size - shard_size + 1)
  if compiler.is_precompiling():
    slice_starts = range(0, full_shard_stop, shard_size)
  else:
    slice_starts = torch.arange(0, full_shard_stop, shard_size)


  output_len = len(slice_starts)
  if last_shard_size != shard_size:
    output_len += 1
  outputs = [None] * output_len

  apply_fun_to_slice = partial(
    _apply_fun_to_slice,
    in_axes,
    fun,
    args,
    kwargs
  )

  if len(slice_starts) > 0:
    for res_i, start_i in enumerate(slice_starts):
      with trace.into(f"{int(start_i)}..{int(start_i) + shard_size - 1}", log=False):
        slice_out = apply_fun_to_slice(start_i, shard_size)
        outputs[res_i] = slice_out
        # Only first run for precompiling
        if compiler.is_precompiling():
          for i in range(res_i + 1, len(outputs)):
            outputs[i] = slice_out
          break

  if last_shard_size != shard_size:
    remainder_start = in_size - last_shard_size
    rem_slice_out = apply_fun_to_slice(remainder_start, last_shard_size)
    outputs[-1] = rem_slice_out
    # outputs = compute_shard(outputs, remainder_start, last_shard_size)

  merge = partial(
    _merge, out_axes
  )

  outputs = tree.map(merge, *outputs)

  return outputs


def sharded_apply(
    fun: Callable[..., PytreeTensor],
    shard_size: int | None = 1,
    in_axes: int = 0,
    out_axes: int = 0,
    new_out_axes: bool = False,
    enabled: bool | None = None,
) -> Callable[..., PytreeTensor]:
  """Sharded apply.

  Applies `fun` over shards to axes, in a way similar to vmap,
  but does so in shards of `shard_size`. Shards are stacked after.
  This allows a smooth trade-off between
  memory usage (as in a plain map) vs higher throughput (as in a vmap).

  Args:
    fun: Function to apply smap transform to.
    shard_size: Integer denoting shard size.
    in_axes: Either integer or pytree describing which axis to map over for each
      input to `fun`, None denotes broadcasting.
    out_axes: Integer or pytree denoting to what axis in the output the mapped
      over axis maps.
    new_out_axes: Whether to stack outputs on new axes. This assumes that the
      output sizes for each shard (including the possible remainder shard) are
      the same.

  Returns:
    Function with smap applied.
  """
  if new_out_axes:
    raise NotImplementedError("New output axes not yet implemented.")

  # shard size None denotes no sharding
  # disable sharding if config.enable_sharding is False
  if shard_size is None or (enabled != True and (not config.enable_sharding or enabled == False)):
    return fun

  mapped_fn = partial(
    _mapped_fn,
    fun,
    in_axes,
    out_axes,
    shard_size
  )

  return mapped_fn


def reshape_partitioned_inputs(
    batched_args: Sequence[PytreeTensor],
    partitioned_dim: int,
    subbatch_size: int,
) -> Sequence[PytreeTensor]:
  """Reshapes so subbatching doesn't happen on the partitioned dim."""
  subbatched_args = []
  for arg in batched_args:
    shape = arg.shape

    new_shape = (
        shape[:partitioned_dim]
        + (subbatch_size, shape[partitioned_dim] // subbatch_size)
        + shape[partitioned_dim + 1:]
    )
    subbatched_args.append(arg.reshape(new_shape))
  return subbatched_args


def reshape_partitioned_output(
    output: torch.Tensor, output_subbatch_dim: int
) -> torch.Tensor:
  """Reshapes outputs as if reshape_partitioned_inputs were never applied."""
  out_shape = (
      output.shape[: output_subbatch_dim - 1]
      + (-1,)
      + output.shape[output_subbatch_dim + 1:]
  )
  return output.reshape(out_shape)


def _run_module(
    input_subbatch_dim_is_partitioned: bool,
    module: Callable[..., PytreeTensor],
    input_subbatch_dim: int,
    output_subbatch_dim: int,
    nonbatched_args: Sequence[PytreeTensor],
    batched_args: Sequence[PytreeTensor]
):
  if input_subbatch_dim_is_partitioned:
    # Squeeze off the singleton dimension (otherwise the module would see
    # [..., subbatch_size, 1, ...]).
    batched_args = [torch.squeeze(b, dim=input_subbatch_dim) for b in batched_args]
  args = list(batched_args) + list(nonbatched_args)
  res = module(*args)
  if input_subbatch_dim_is_partitioned:
    # Add back in the singleton dimension so the outputs are stacked on the
    # axis we are actually subbatching over (i.e stacked back to
    # [..., subbatch_size, partitioned_input_size // subbatch_size, ...]),
    # rather than on the partitioned axis, which would again induce an
    # all-gather that breaks partitioning.
    res = torch.unsqueeze(res, dim=output_subbatch_dim)
  return res


def inference_subbatch(
    module: Callable[..., PytreeTensor],
    subbatch_size: int,
    batched_args: Sequence[PytreeTensor],
    nonbatched_args: Sequence[PytreeTensor],
    input_subbatch_dim: int = 0,
    output_subbatch_dim: int | None = None,
    input_subbatch_dim_is_partitioned: bool = False,
) -> PytreeTensor:
  """Run through subbatches (like batch apply but with split and concat)."""
  assert len(batched_args) > 0  # pylint: disable=g-explicit-length-test

  if output_subbatch_dim is None:
    output_subbatch_dim = input_subbatch_dim

  if input_subbatch_dim_is_partitioned:
    # Subbatching along the partitioned axis would induce an all-gather that
    # undoes the partitioning. So instead we reshape such that
    # [..., partitioned_input_size, ...] becomes [..., subbatch_size,
    # partitioned_input_size // subbatch_size, ...] and then actually subbatch
    # along the partitioned_input_size // subbatch_size axis in slices of
    # size 1. Partitioning is then preserved on the partitioned axis, except
    # that dimension is now of size subbatch_size instead of
    # partitioned_input_size. Note that the module itself still sees inputs of
    # size [..., subbatch_size, ...], just as it would if this reshaping were
    # not applied.
    batched_args = reshape_partitioned_inputs(
      batched_args, input_subbatch_dim, subbatch_size
    )
    input_subbatch_dim += 1
    output_subbatch_dim += 1
    subbatch_size = 1

  run_module = partial(
    _run_module,
    input_subbatch_dim_is_partitioned,
    module,
    input_subbatch_dim,
    output_subbatch_dim,
    nonbatched_args
  )

  sharded_module = sharded_apply(
    run_module,
    shard_size=subbatch_size,
    in_axes=input_subbatch_dim,
    out_axes=output_subbatch_dim,
  )
  output = sharded_module(batched_args)
  if input_subbatch_dim_is_partitioned:
    # The is of the same shape as the inputs [..., subbatch_size,
    # partitioned_input_size // subbatch_size, ...]. Reshape to
    # [..., partitioned_input_size, ...] as if the reshaping due to partitioning
    # had never been applied.
    output = reshape_partitioned_output(output, output_subbatch_dim)

  return output
