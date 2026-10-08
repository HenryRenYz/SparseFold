# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from typing import Any, Self, TypeVar
import os
import functools
import io
import math
import pickle
import time
import resource
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as _dist
from torch.futures import Future

from sparsefold.runtime import trace

def is_enabled() -> bool:
  return _curr_backend is not None


def is_primary() -> bool:
  """True if is on primary process (rank 0), or not initialize distributed."""
  if is_enabled():
    return get_rank() == 0
  else:
    return True

def init(
    rank: int | None = None,
    world_size: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    backend: str = "gloo",
    **kwargs
): 
  if rank is None:
    rank = int(os.environ["RANK"])
  if world_size is None:
    world_size = int(os.environ["WORLD_SIZE"])
  if master_addr is None:
    master_addr = os.environ.get("MASTER_ADDR", None)
  if master_port is None:
    master_port = os.environ.get("MASTER_PORT", None)
    if master_port is not None:
      master_port = int(master_port)

  _init_backend(
      backend,
      rank=rank,
      world_size=world_size,
      master_addr=master_addr,
      master_port=master_port,
      **kwargs
  )

_Return = TypeVar("_Return")
_FutureOrValue = Future[_Return] | _Return


class _DeferredFuture:
  """Minimal wait/then future for process groups without Work.get_future()."""

  def __init__(self, wait_fn):
    self._wait_fn = wait_fn
    self._done = False
    self._value = None

  def wait(self):
    if not self._done:
      self._value = self._wait_fn()
      self._done = True
    return self._value

  def value(self):
    return self.wait()

  def then(self, callback):
    return _DeferredFuture(lambda: callback(self))

def get_rank() -> int:
  return _get_backend().get_rank()
  

def get_world_size() -> int:
  return _get_backend().get_world_size()


def barrier() -> None:
  _get_backend().barrier()


def broadcast(
    tensor: torch.Tensor,
    src: int | None = None,
    async_op: bool = False,
) -> _FutureOrValue[torch.Tensor]: 
  return _get_backend().broadcast(tensor, src, async_op)


def broadcast_object_list(
    object_list: list[Any],
    src: int | None = None,
    device: torch.device | None = None,
) -> _FutureOrValue[list[Any]]:
  return _get_backend().broadcast_object_list(object_list, src, device)

def all_gather(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    async_op: bool = False,
) -> _FutureOrValue[list[torch.Tensor]]: 
  return _get_backend().all_gather(tensor_list, tensor, async_op)


def all_gather_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    async_op: bool = False
) -> _FutureOrValue[torch.Tensor]:
  """
  All `input_tensor` should be the same size on each process.
  """
  return _get_backend().all_gather_into_tensor(output_tensor, input_tensor, async_op)


def all_gatherv_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    output_split_sizes: list[int],
    async_op: bool = False,
) -> _FutureOrValue[torch.Tensor]:
  """Gather rank-varying dim-0 shards into one contiguous tensor."""
  return _get_backend().all_gatherv_into_tensor(
      output_tensor, input_tensor, output_split_sizes, async_op
  )


def scatter(
    tensor: torch.Tensor,
    scatter_list: list[torch.Tensor] | None = None,
    src: int | None = None,
    async_op: bool = False,
) -> _FutureOrValue[torch.Tensor]: 
  """
  All tensors in tensor_list must have the same size.
  """
  return _get_backend().scatter(tensor, scatter_list, src, async_op)


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
    async_op: bool = False,
) -> _FutureOrValue[torch.Tensor]:
  return _get_backend().all_to_all_single(output, input, output_split_sizes, input_split_sizes, async_op)


def all_to_all(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    async_op: bool = False
) -> _FutureOrValue[list[torch.Tensor]]:
  return _get_backend().all_to_all(output_tensor_list, input_tensor_list, async_op)



class Shard:
  def __init__(self, length: int, num_shards: int | None = None, even: bool = True):
    if not is_enabled():
      raise AssertionError("Call `init` first before creating Shard")
    if num_shards is None:
      num_shards = get_world_size()
    else:
      assert num_shards <= get_world_size()
    if even:
      assert num_shards == get_world_size(), "In even partition mode, `num_shards` should be consistent with world_size"
      assert length % num_shards == 0, "In even partition mode, `length` should be divisible by world_size"
    
    self.length = length
    self.num_shards = num_shards
    self.is_even = even
    # always length `world_size`
    self._offsets: list[int] = []
    self._sizes: list[int] = []
    self._compute_offsets_and_sizes()

  def size(self, rank: int | None = None) -> int:
    if rank is None:
      rank = get_rank()
    assert 0 <= rank < get_world_size()
    return self._sizes[rank]

  def offset(self, rank: int | None = None) -> int:
    if rank is None:
      rank = get_rank()
    assert 0 <= rank < get_world_size()
    return self._offsets[rank]
  
  def sizes(self) -> list[int]:
    return self._sizes
  
  def offsets(self) -> list[int]:
    return self._offsets
  
  def _compute_offsets_and_sizes(self) -> tuple[list[int], list[int]]:
    if self.is_even:
      single_size = self.length // self.num_shards
      assert self.num_shards == get_world_size()
      sizes = [single_size] * self.num_shards
    elif self.length <= self.num_shards:
      sizes = [1] * self.length + [0] * (get_world_size() - self.length)
    else:
      sizes = []
      single_size_f = self.length / self.num_shards
      offset = 0
      for i in range(self.num_shards):
        end_offset = int(round((i + 1) * single_size_f))
        size = end_offset - offset
        sizes.append(size)
        offset = end_offset
      sizes += [0] * (get_world_size() - self.num_shards)
      max_at = np.argmax(sizes)
      # Move longest elements to rank 0
      # to guarantee that rank 0 always run the longest time
      sizes[0], sizes[max_at] = sizes[max_at], sizes[0]
    offsets = [0] + [int(x) for x in np.cumsum(sizes)[:-1]]
    self._sizes = sizes
    self._offsets = offsets
    return offsets, sizes

  def broadcast(
      self,
      *,
      tensor: torch.Tensor,
      src: int | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]: 
    assert self.is_even, "`broadcast` op only applicable to even partitioning"
    if src == get_rank():
      tensor = _make_contiguous(tensor, "broadcast_input")
    return broadcast(tensor, src, async_op)

  def all_gather(
      self,
      *,
      tensor_list: list[torch.Tensor] | None = None,
      tensor: torch.Tensor,
      async_op: bool = False,
  ) -> _FutureOrValue[list[torch.Tensor]]: 
    if tensor_list is None:
      tensor_list = [torch.empty(
          (self.size(i),) + tensor.shape[1:], 
          dtype=tensor.dtype, layout=tensor.layout, 
          device=tensor.device
      ) for i in range(get_world_size())]
    assert tensor_list[get_rank()].shape == tensor.shape
    tensor = _make_contiguous(tensor, "all_gather_input")
    if self.is_even:
      return all_gather(tensor_list, tensor, async_op)
    else:
      if _curr_backend.name == "gloo":
        world_size = get_world_size()

        output_split_sizes = self.sizes()
        total_length = sum(output_split_sizes)
        output_flattened = torch.empty(
            (total_length,) + tensor.shape[1:],
            dtype=tensor.dtype, device=tensor.device
        )
        
        input_split_sizes = [tensor.shape[0]] * world_size
        repeat_shape = [1] * tensor.ndim
        repeat_shape[0] = world_size
        input_flattened = tensor.repeat(*repeat_shape)
        
        work = self.all_to_all_single(
            output=output_flattened,
            input=input_flattened,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            async_op=async_op
        )
        
        def unpack_result(output_tensor: torch.Tensor):
          offset = 0
          for i, size in enumerate(output_split_sizes):
              tensor_list[i].copy_(output_tensor[offset : offset + size])
              offset += size
          return tensor_list

        if async_op:
          return work.then(lambda completed: unpack_result(completed.value()))
        else:
          return unpack_result(output_flattened)
      else:
        input_tensor_list = [tensor] * get_world_size()
        return self.all_to_all(
            output_tensor_list=tensor_list,
            input_tensor_list=input_tensor_list,
            async_op=async_op
        )

  def all_gather_into_tensor(
      self,
      *,
      output_tensor: torch.Tensor | None = None,
      input_tensor: torch.Tensor,
      dim: int = 0,
      async_op: bool = False
  ) -> _FutureOrValue[torch.Tensor]:
    if dim < 0:
      dim += input_tensor.ndim
    input_permute_dims, output_permute_dims = _get_movefront_permute(input_tensor.ndim, [dim])
    
    if output_tensor is None:
      output_shape = (self.length,) + input_tensor.shape[:dim] + input_tensor.shape[dim+1:]
      output_tensor = torch.empty(
          output_shape,
          dtype=input_tensor.dtype, layout=input_tensor.layout,
          device=input_tensor.device
      )
    else:
      expected_output_shape = input_tensor.shape[:dim] + (self.length,) + input_tensor.shape[dim+1:]
      assert output_tensor.shape == expected_output_shape, f"Expected output shape {tuple(expected_output_shape)}, but got {tuple(output_tensor.shape)}"
      output_tensor = torch.permute(output_tensor, input_permute_dims)

    input_tensor = torch.permute(input_tensor, input_permute_dims)
    input_tensor = _make_contiguous(input_tensor, "all_gather_into_tensor_input")
    # Note: some backends requires output_tensor be contiguous
    output_tensor = _make_contiguous(output_tensor, "all_gather_into_tensor_output")

    if self.is_even:
      return _decorate_future(
          all_gather_into_tensor(output_tensor, input_tensor, async_op),
          on_return=lambda x: torch.permute(x, output_permute_dims)
      )
    else:
      return _decorate_future(
          all_gatherv_into_tensor(
              output_tensor, input_tensor, self.sizes(), async_op
          ),
          on_return=lambda x: torch.permute(x, output_permute_dims),
      )

  def all_gather_padded_into_tensor(
      self,
      *,
      input_tensor: torch.Tensor,
      dim: int = 0,
  ) -> torch.Tensor:
    """Gather an uneven shard through the regular equal-size collective.

    Every rank pads its local shard to the largest rank size, participates in
    ``all_gather_into_tensor``, then removes the padding locally.  This is a
    deliberately synchronous completion path for small ragged result trees;
    it avoids introducing a distinct AllGatherV collective after a long model
    run made entirely of regular collectives.
    """
    if dim < 0:
      dim += input_tensor.ndim
    if not 0 <= dim < input_tensor.ndim:
      raise IndexError(
          f"gather dimension {dim} out of range for rank-{input_tensor.ndim} tensor"
      )

    sizes = self.sizes()
    local_size = sizes[get_rank()]
    if input_tensor.shape[dim] != local_size:
      raise ValueError(
          f"local shard dimension is {input_tensor.shape[dim]}, expected {local_size}"
      )

    # FakeTensor/Meta precompilation is shape propagation only.  All ranks run
    # the same trace, but no payload exists to communicate; return a metadata
    # tensor with the final gathered extent without crossing the backend.
    metadata_only = input_tensor.is_meta
    if not metadata_only:
      try:
        from torch._subclasses.fake_tensor import is_fake
      except ImportError:  # pragma: no cover - older Torch compatibility.
        pass
      else:
        metadata_only = bool(is_fake(input_tensor))
    if metadata_only:
      output_shape = list(input_tensor.shape)
      output_shape[dim] = self.length
      return input_tensor.new_empty(output_shape)

    max_size = max(sizes, default=0)
    if max_size == 0:
      return input_tensor.narrow(dim, 0, 0)

    input_permute_dims, output_permute_dims = _get_movefront_permute(
        input_tensor.ndim, [dim]
    )
    local = torch.permute(input_tensor, input_permute_dims).contiguous()
    padded = torch.zeros(
        (max_size,) + tuple(local.shape[1:]),
        dtype=local.dtype,
        device=local.device,
    )
    if local_size:
      padded[:local_size].copy_(local)

    gathered = torch.empty(
        (get_world_size() * max_size,) + tuple(local.shape[1:]),
        dtype=local.dtype,
        device=local.device,
    )
    all_gather_into_tensor(gathered, padded)
    rank_major = gathered.reshape(
        (get_world_size(), max_size) + tuple(local.shape[1:])
    )
    unpadded = torch.cat(
        [rank_major[rank, :size] for rank, size in enumerate(sizes)], dim=0
    )
    return torch.permute(unpadded, output_permute_dims)

  def scatter(
      self,
      *,
      tensor: torch.Tensor,
      scatter_list: list[torch.Tensor] | None = None,
      src: int | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]: 
    """
    Even partition mode only
    All tensors in tensor_list must have the same size.
    """
    assert self.is_even, "`scatter` op only applied to even partitioning"
    if scatter_list is not None:
      scatter_list = [_make_contiguous(t, f"scatter_input_{i}") for i, t in enumerate(scatter_list)]
    return scatter(tensor, scatter_list, src, async_op)
    
  def all_to_all_single(
      self,
      *,
      output: torch.Tensor,
      input: torch.Tensor,
      output_split_sizes: list[int] | None = None,
      input_split_sizes: list[int] | None = None,
      output_dim: int = 0,
      input_dim: int = 0,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]:
    output_fwd_permute_dims, output_rev_permute_dims = _get_movefront_permute(output.ndim, [output_dim])
    input_fwd_permute_dims, _ = _get_movefront_permute(input.ndim, [input_dim])
    input = torch.permute(input, input_fwd_permute_dims)
    output = torch.permute(output, output_fwd_permute_dims)

    output = _make_contiguous(output, "all_to_all_single_output")
    input = _make_contiguous(input, "all_to_all_single_input")

    return _decorate_future(
        all_to_all_single(output, input, output_split_sizes, input_split_sizes, async_op),
        on_return=lambda x: torch.permute(x, output_rev_permute_dims)
    )

  def all_to_all(
      self,
      *,
      output_tensor_list: list[torch.Tensor],
      input_tensor_list: list[torch.Tensor],
      async_op: bool = False
  ) -> _FutureOrValue[list[torch.Tensor]]:
    input_tensor_list = [_make_contiguous(t, f"all_to_all_input_{i}") for i, t in enumerate(input_tensor_list)]

    return all_to_all(output_tensor_list, input_tensor_list, async_op)
  
  def reshard(
      self,
      *,
      output: torch.Tensor | None = None,
      input: torch. Tensor,
      output_shard: "Shard | None" = None,
      dim1: int = 0,
      dim2: int = 1,
      async_op: bool = False
  ) -> _FutureOrValue[torch.Tensor]:
    """
    dim1: shard dimension, input.shape[dim1] == self.size()
    dim2: target dimension to reshard, input.shape[dim2] == self.length
    output_shard: another shard, or self if not specified
    """
    if output_shard is None:
      output_shard = self
    
    # Move `dim1` to dim 0, `dim2` to dim 1, for `input`
    if dim1 < 0:
      dim1 += input.ndim
    if dim2 < 0:
      dim2 += input.ndim

    assert dim1 != dim2
    assert input.ndim >= 2
    assert input.shape[dim1] == self.size(get_rank()) and input.shape[dim2] == output_shard.length
    input_permute_dims, output_permute_dims = _get_movefront_permute(input.ndim, [dim1, dim2])
    input = torch.permute(input, input_permute_dims)

    # Convert [local_dim1, global_dim2, ...] to rank-major blocks.  For an
    # even target shard this is one strided copy instead of world_size slice
    # views followed by a Python-level cat.  Keep the ragged fallback for
    # uneven partitions, whose blocks cannot be represented by one reshape.
    if output_shard.is_even:
      target_size = output_shard.size()
      send_tensor = (
          input.unflatten(1, (get_world_size(), target_size))
          .movedim(1, 0)
          .contiguous()
          .flatten(start_dim=0, end_dim=2)
      )
    else:
      input_split = torch.split(input, output_shard.sizes(), dim=1)
      send_tensor = torch.cat(
          [t.flatten(start_dim=0, end_dim=1) for t in input_split]
      ).contiguous()
    send_split_sizes = [self.size(get_rank()) * output_shard.size(i) for i in range(get_world_size())]
    recv_split_sizes = [self.size(i) * output_shard.size(get_rank()) for i in range(get_world_size())]
    total_recv_els = sum(recv_split_sizes)
    recv_tensor = torch.empty(
        (total_recv_els,) + input.shape[2:],
        dtype=input.dtype,
        device=input.device,
    )
    # recv_tensor is empty -> contiguous
    # send_tensor is cat result -> contiguous
    work = all_to_all_single(recv_tensor, send_tensor, recv_split_sizes, send_split_sizes, async_op=async_op)

    def write_back(out_tensor: torch.Tensor):
      nonlocal output
      target_shard_size = output_shard.size(get_rank())

      # all_to_all_single concatenates source-rank blocks in rank order.  The
      # blocks therefore form the complete dim1 directly; no per-rank split
      # and indexed copy is needed on the receive path.
      reshaped = out_tensor.view(
          (self.length, target_shard_size) + input.shape[2:]
      )
      reshaped = torch.permute(reshaped, output_permute_dims)

      if output is None:
        # In the common dim1=0/dim2=1 case this returns the receive buffer
        # itself.  Preserve the old contiguous-output contract for other
        # dimension orders with one copy rather than world_size copies.
        return reshaped if reshaped.is_contiguous() else reshaped.contiguous()

      assert (
          output.shape[dim2] == target_shard_size
          and output.shape[dim1] == self.length
      )
      output.copy_(reshaped)
      return output

    if async_op:
      return work.then(lambda completed: write_back(completed.value()))
    else:
      return write_back(recv_tensor)
  
  def shard(
      self,
      tensor: torch.Tensor,
      dim: int = 0
  ) -> torch.Tensor:
    assert tensor.shape[dim] == self.length
    if dim < 0:
      dim += tensor.ndim
    if self.num_shards == 1:
      return tensor
    indices = (slice(None),) * dim + (slice(self.offset(), self.offset() + self.size()),)
    return tensor[indices].contiguous()
  

def _make_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
  if x.is_contiguous():
    return x
  trace.log_debug(f"CCL: tensor `{name}` is not contiguous")
  return x.contiguous()
  
def _get_movefront_permute(ndim: int, dims: list[int]) -> tuple[list[int], list[int]]:
  dims = [(d + ndim if d < 0 else d) for d in dims]
  for i, d in enumerate(dims):
    assert 0 <= d < ndim, f"At index {i}: dimension {d} not within 0..{ndim}"
  assert len(set(dims)) == len(dims), f"All dimensions should be unique in list: {dims}"

  # Compute forward permutation: move dims to front, then the rest in order
  remaining_dims = [d for d in range(ndim) if d not in dims]
  permute_axes = dims + remaining_dims

  # Compute reverse permutation: invert the forward permutation
  reverse_axes = [0] * ndim
  for new_pos, old_pos in enumerate(permute_axes):
    reverse_axes[old_pos] = new_pos

  return permute_axes, reverse_axes
    

current_shard: Shard | None = None

_registered_backends: dict[str, Any] = {}

def register_backend(
    init_fn = None,
    /, *,
    name: str,
):
  if init_fn is None:
    return functools.partial(register_backend, name=name)
  
  if name in _registered_backends:
    raise KeyError(f"Backend `{name}` already registered!")
  
  _registered_backends[name] = init_fn
  return init_fn


_curr_backend: "Backend | None" = None

def _init_backend(
    backend: str,
    *args,
    **kwargs
) -> "Backend":
  global _curr_backend

  if _curr_backend is not None:
    raise AssertionError(f"Backend is already initialized ({_curr_backend})")
  
  if backend not in _registered_backends:
    raise KeyError(f"Backend `{backend}` not registered")
  fn = _registered_backends[backend]
  backend = fn(*args, **kwargs)
  _curr_backend = backend
  return backend
  

def _get_backend() -> "Backend":
  if _curr_backend is None:
    raise ValueError("Backend not initialized, call `init` first")
  return _curr_backend


class Backend:
  name: str

  @classmethod
  @abstractmethod
  def init(
      cls,
      rank: int,
      world_size: int,
      master_addr: str | None,
      master_port: str | None,
      **kwargs
  ) -> Self: 
    ...
  
  @abstractmethod
  def get_rank(self) -> int:
    ...

  @abstractmethod
  def get_world_size(self) -> int:
    ...

  @abstractmethod
  def barrier(self) -> None:
    ...

  @abstractmethod
  def broadcast(
      self,
      tensor: torch.Tensor,
      src: int | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]: 
    ...

  @abstractmethod
  def all_gather(
      self,
      tensor_list: list[torch.Tensor],
      tensor: torch.Tensor,
      async_op: bool = False,
  ) -> _FutureOrValue[list[torch.Tensor]]: 
    ...

  def all_gather_into_tensor(
      self,
      output_tensor: torch.Tensor,
      input_tensor: torch.Tensor,
      async_op: bool = False
  ) -> _FutureOrValue[torch.Tensor]:
    """
    All `input_tensor` should be the same size on each process.
    """
    tensors = [torch.empty_like(input_tensor) for r in range(self.get_world_size())]
    work = self.all_gather(tensors, input_tensor, async_op=async_op)
    
    def write_back(_=None):
      torch.cat(tensors, out=output_tensor)

    if async_op:
      work.add_done_callback(write_back)
      return work
    else:
      write_back()
      return

  def all_gatherv_into_tensor(
      self,
      output_tensor: torch.Tensor,
      input_tensor: torch.Tensor,
      output_split_sizes: list[int],
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]:
    """Portable ragged AllGather fallback using the tensor-list operation."""
    if len(output_split_sizes) != self.get_world_size():
      raise ValueError("output_split_sizes must have world_size entries")
    if sum(output_split_sizes) != output_tensor.shape[0]:
      raise ValueError("output_split_sizes must sum to output_tensor.shape[0]")
    tensor_list = list(torch.split(output_tensor, output_split_sizes, dim=0))
    return _decorate_future(
        self.all_gather(tensor_list, input_tensor, async_op), output_tensor
    )

  @abstractmethod
  def scatter(
      self,
      tensor: torch.Tensor,
      scatter_list: list[torch.Tensor] | None = None,
      src: int | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]: 
    """
    All tensors in tensor_list must have the same size.
    """
    ...

  @abstractmethod
  def all_to_all_single(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      output_split_sizes: list[int] | None = None,
      input_split_sizes: list[int] | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]:
    ...

  @abstractmethod
  def all_to_all(
      self,
      output_tensor_list: list[torch.Tensor],
      input_tensor_list: list[torch.Tensor],
      async_op: bool = False
  ) -> _FutureOrValue[list[torch.Tensor]]:
    ...

  @abstractmethod
  def broadcast_object_list(
      self,
      object_list: list[Any],
      src: int | None = None,
      device: torch.device | None = None
  ) -> _FutureOrValue[list[Any]]:
      """
      All objects in `object_list` should be pickable.
      """
      ...


def _decorate_future(x: _Return, output: _Return=None, on_return=lambda t: t) -> _FutureOrValue[_Return]:
  if x is None:
    x = output
  if isinstance(x, _DeferredFuture):
    return x.then(
        lambda completed: on_return(
            output if output is not None else completed.value()
        )
    )
  if isinstance(x, _dist.Work):
    try:
      x = x.get_future()
    except RuntimeError as error:
      if "getFuture not implemented" not in str(error):
        raise
      work = x

      def wait_work():
        result = work.wait()
        return on_return(output if output is not None else result)

      return _DeferredFuture(wait_work)
  if isinstance(x, (Future, torch.Future)):
    return x.then(
        lambda completed: on_return(
            output if output is not None else completed.value()
        )
    )
  else:
    return on_return(x)

class TorchDistBackend(Backend):
  def __init__(
      self,
      backend: str
  ):
    self.name = backend

  @classmethod
  def init(
      cls,
      backend: str,
      rank: int,
      world_size: int,
      master_addr: str | None,
      master_port: int | None,
      **kwargs
  ) -> Self: 
    backend_obj: TorchDistBackend = cls(backend)
    if master_addr is not None:
      assert master_port is not None
      init_method = f"tcp://{master_addr}:{master_port}"
    else:
      assert master_port is None
      init_method = "env://"
    init_kwargs = dict(
        backend=backend_obj.name,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    if backend == "kutacc":
      # The transport timeout is extended for rank-0 vecops compilation.
      # Keep PyTorch's process-group timeout above it so PyTorch does not
      # abort a healthy compile while the other ranks wait at the barrier.
      wait_ms = int(os.environ.get("KUTACC_SHM_WAIT_TIMEOUT_MS", "120000"))
      if wait_ms < 0:
        raise ValueError("KUTACC_SHM_WAIT_TIMEOUT_MS must be non-negative")
      if wait_ms:
        init_kwargs["timeout"] = timedelta(milliseconds=wait_ms + 60000)
    _dist.init_process_group(**init_kwargs)
    return backend_obj
    
  
  def get_rank(self) -> int:
    return _dist.get_rank()

  def get_world_size(self) -> int:
    return _dist.get_world_size()

  def barrier(self) -> None:
    self._profile_collective("barrier", (), lambda: _dist.barrier())

  @staticmethod
  def _profile_collective(name: str, shape, fn):
    if os.environ.get("SPARSEFOLD_PROFILE_DIST", os.environ.get("AF3_PROFILE_DIST")) != "1":
      return fn()
    rank = _dist.get_rank()
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter_ns()
    try:
      return fn()
    finally:
      elapsed = (time.perf_counter_ns() - started) / 1e6
      usage1 = resource.getrusage(resource.RUSAGE_SELF)
      path = trace.current_path()
      print(
          f"[dist-profile rank={rank} backend=kutacc op={name} "
          f"path={path} t_ns={time.monotonic_ns()} "
          f"shape={tuple(shape)} elapsed_ms={elapsed:.3f} "
          f"nvcsw={usage1.ru_nvcsw - usage0.ru_nvcsw} "
          f"nivcsw={usage1.ru_nivcsw - usage0.ru_nivcsw}]",
          flush=True,
      )

  def broadcast(
      self,
      tensor: torch.Tensor,
      src: int | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]: 
    return _decorate_future(
        self._profile_collective(
            "broadcast", tensor.shape,
            lambda: _dist.broadcast(tensor, src, async_op=async_op),
        ), tensor
    )

  def all_gather(
      self,
      tensor_list: list[torch.Tensor],
      tensor: torch.Tensor,
      async_op: bool = False,
  ) -> _FutureOrValue[list[torch.Tensor]]: 
    return _decorate_future(
        self._profile_collective(
            "all_gather", tensor.shape,
            lambda: _dist.all_gather(tensor_list, tensor, async_op=async_op),
        ), tensor_list
    )

  def all_gather_into_tensor(
      self,
      output_tensor: torch.Tensor,
      input_tensor: torch.Tensor,
      async_op: bool = False
  ) -> _FutureOrValue[torch.Tensor]:
    return _decorate_future(
        self._profile_collective(
            "all_gather_into_tensor", input_tensor.shape,
            lambda: _dist.all_gather_into_tensor(
                output_tensor, input_tensor, async_op=async_op
            ),
        ), output_tensor
    )

  def all_gatherv_into_tensor(
      self,
      output_tensor: torch.Tensor,
      input_tensor: torch.Tensor,
      output_split_sizes: list[int],
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]:
    if len(output_split_sizes) != self.get_world_size():
      raise ValueError("output_split_sizes must have world_size entries")
    if sum(output_split_sizes) != output_tensor.shape[0]:
      raise ValueError("output_split_sizes must sum to output_tensor.shape[0]")

    if self.name == "kutacc":
      import kutacc_for_torch

      elements_per_row = math.prod(input_tensor.shape[1:])
      element_counts = [
          rows * elements_per_row for rows in output_split_sizes
      ]
      return _decorate_future(
          kutacc_for_torch.all_gatherv_into_tensor(
              output_tensor,
              input_tensor,
              element_counts,
              async_op=async_op,
          ),
          output_tensor,
      )

    if self.name == "gloo":
      # Gloo does not accept uneven tensors in its list AllGather.  Express
      # the gather as an AllToAllV while keeping this workaround inside the
      # backend adapter rather than leaking it into the Shard abstraction.
      world_size = self.get_world_size()
      input_split_sizes = [input_tensor.shape[0]] * world_size
      repeat_shape = [1] * input_tensor.ndim
      repeat_shape[0] = world_size
      input_flattened = input_tensor.repeat(*repeat_shape)
      return self.all_to_all_single(
          output_tensor,
          input_flattened,
          output_split_sizes,
          input_split_sizes,
          async_op,
      )

    return super().all_gatherv_into_tensor(
        output_tensor, input_tensor, output_split_sizes, async_op
    )

  def scatter(
      self,
      tensor: torch.Tensor,
      scatter_list: list[torch.Tensor] | None = None,
      src: int | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]: 
    return _decorate_future(
        _dist.scatter(tensor, scatter_list, src, async_op=async_op), tensor
    )

  def all_to_all_single(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      output_split_sizes: list[int] | None = None,
      input_split_sizes: list[int] | None = None,
      async_op: bool = False,
  ) -> _FutureOrValue[torch.Tensor]:
    return _decorate_future(
        self._profile_collective(
            "all_to_all_single", input.shape,
            lambda: _dist.all_to_all_single(
                output, input, output_split_sizes, input_split_sizes,
                async_op=async_op,
            ),
        ), output
    )

  def all_to_all(
      self,
      output_tensor_list: list[torch.Tensor],
      input_tensor_list: list[torch.Tensor],
      async_op: bool = False
  ) -> _FutureOrValue[list[torch.Tensor]]:
    shape = input_tensor_list[0].shape if input_tensor_list else ()
    return _decorate_future(
        self._profile_collective(
            "all_to_all", shape,
            lambda: _dist.all_to_all(
                output_tensor_list, input_tensor_list, async_op=async_op
            ),
        ), output_tensor_list
    )
  
  def broadcast_object_list(
      self,
      object_list: list[Any],
      src: int | None = None,
      device: torch.device | None = None
  ) -> _FutureOrValue[list[Any]]:
    if self.name == "kutacc":
      # KUPL's contiguous window allocation is limited to INT_MAX bytes after
      # summing every rank's contribution. PyTorch's stock helper serializes
      # the complete object list into one tensor, which exceeds that limit for
      # AF3's state_dict. Keep the backend-specific constraint here and stream
      # the same pickle through bounded native broadcasts.
      if device is not None and torch.device(device).type != "cpu":
        raise ValueError("The Kutacc backend only supports CPU object broadcast")
      if src is None:
        src = 0

      serialized: io.BytesIO | None = None
      if self.get_rank() == src:
        serialized = io.BytesIO()
        pickle.Pickler(serialized, protocol=pickle.HIGHEST_PROTOCOL).dump(
            object_list
        )
        payload = serialized.getbuffer()
        payload_size = len(payload)
      else:
        payload = None
        payload_size = 0

      size_tensor = torch.tensor([payload_size], dtype=torch.int64)
      self.broadcast(size_tensor, src=src)
      payload_size = int(size_tensor.item())
      if self.get_rank() != src:
        payload = memoryview(bytearray(payload_size))

      chunk_bytes = int(os.environ.get(
          "SPARSEFOLD_KUTACC_OBJECT_BROADCAST_CHUNK_BYTES", os.environ.get("AF3_KUTACC_OBJECT_BROADCAST_CHUNK_BYTES", 64 * 1024 * 1024)
      ))
      if chunk_bytes <= 0:
        raise ValueError(
            "SPARSEFOLD_KUTACC_OBJECT_BROADCAST_CHUNK_BYTES must be positive"
        )
      assert payload is not None
      for offset in range(0, payload_size, chunk_bytes):
        chunk = torch.frombuffer(
            payload[offset:min(offset + chunk_bytes, payload_size)],
            dtype=torch.uint8,
        )
        self.broadcast(chunk, src=src)

      if self.get_rank() != src:
        received = pickle.loads(payload)
        if not isinstance(received, list):
          raise TypeError("Kutacc object broadcast payload is not a list")
        object_list[:] = received
      # Object deserialization is rank-local.  Without a completion barrier,
      # the primary rank can enter model compute while another rank is still
      # unpickling the same payload; the next tensor collective then reports a
      # multi-second "operator" tail even though its transfer is short.
      self.barrier()
      return object_list

    return _decorate_future(
        _dist.broadcast_object_list(object_list, src, device=device), object_list
    )


# Default backends provided by PyTorch
register_backend(functools.partial(TorchDistBackend.init, backend="gloo"), name="gloo")
register_backend(functools.partial(TorchDistBackend.init, backend="mpi"), name="mpi")
register_backend(functools.partial(TorchDistBackend.init, backend="nccl"), name="nccl")
register_backend(functools.partial(TorchDistBackend.init, backend="xccl"), name="xccl")


def _init_kutacc_backend(**kwargs) -> TorchDistBackend:
  try:
    import kutacc_for_torch  # noqa: F401
  except ImportError as error:
    raise RuntimeError(
        "The 'kutacc' backend requires the optional kutacc-for-torch package"
    ) from error
  return TorchDistBackend.init(backend="kutacc", **kwargs)


register_backend(_init_kutacc_backend, name="kutacc")
