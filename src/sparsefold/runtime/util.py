# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import hashlib
import inspect
import functools
import sys
import math
import contextlib
from types import FunctionType
from typing import Sequence, Final, Set, Mapping

import numpy as np
import torch
from torch.nn.utils import stateless

from sparsefold.runtime import config
from sparsefold.runtime import compiler


Tensor = torch.Tensor
TensorLike = Tensor | np.ndarray
Shape = Sequence[int]
Slice = Sequence[int | torch.LongTensor] | int | torch.LongTensor

# DEBUG: Final[bool] = int(os.environ.get("AF3_DEBUG", 0)) == 1
DEBUG = True
DEFAULT_FP32_MATMUL_PRECISION: Final[str] = "medium"
DEFAULT_DEVICE: str = config.device

_TORCH_INTEGERS: Final[Set[torch.dtype]] = {
  torch.int8,
  torch.int16,
  torch.int32,
  torch.int64,
  torch.uint8,
  torch.uint16,
  torch.uint32,
  torch.uint64,
}

_TORCH_DTYPE_MAP: Mapping[str, torch.dtype] = {
  "fp32": torch.float32,
  "fp64": torch.float64,
  "fp16": torch.float16,
  "bf16": torch.bfloat16,
}

def slice_tensor(*args, slice: Slice):
  if not isinstance(slice, Sequence):
    slice = (slice,)
    
  assert len(args) > 0
  
  out_arrays = []
  for array in args:
    for x in slice:
      array = dynamic_select(array, x)
    out_arrays.append(array)
  
  if len(out_arrays) == 1:
    return out_arrays[0]
  else:
    return tuple(out_arrays)

def slice_module(mod: torch.nn.Module, slice: Slice):
  from sparsefold.runtime import tree
  state_dict = mod.state_dict()
  state_dict = tree.map(lambda x: slice_tensor(x, slice=slice), state_dict)
  return stateless._reparametrize_module(mod, state_dict)


def is_integer(dtype: torch.dtype) -> bool:
  return dtype in _TORCH_INTEGERS


def get_dtype(name: str) -> torch.dtype:
  return _TORCH_DTYPE_MAP[name]


def bitwise_right_shift_logical(x: Tensor, y: Tensor | int) -> Tensor:
  mask = (1 << (y - x)) - 1
  return (x >> y) & mask


def dynamic_select(array: torch.Tensor, x):
  if isinstance(x, int):
    return array.select(dim=0, index=x)
  elif isinstance(x, torch.Tensor):
    if torch.compiler.is_compiling() or compiler.is_precompiling():
      return torch.index_select(
        array, dim=0, index=x.to(device=array.device)
      ).squeeze(0)
    else:
      assert x.ndim == 0 or x.ndim == 1
      w = x.item()
      torch.sym_constrain_range(w, min=0, max=array.shape[0]-1)
      return array.select(dim=0, index=w)
  else:
    raise ValueError(f"Invalid index type: {type(x)}")


def align_batch(x: torch.Tensor, batch: Sequence[int]):
  shape = x.shape
  x = x.view((1,) * len(batch) + shape)
  x = x.broadcast_to(batch + shape)
  return x


def vmap_slice(
    array: torch.Tensor,
    index: torch.Tensor,
    *,
    start_dim=0,
    ndim=1,  # TODO buggy, do not USE
):
  r"""
  Perform a vmapped slicing.

  Args:
    array - the array to be sliced
    index - the index used to make slice
    start_dim - the starting dimension to make slice in `array`
    ndim - number of vmapped dimensions (starting at 'start_dim')

  Returns:
    the sliced tensor
  """
  # Constraint check
  if start_dim < 0:
    start_dim += array.ndim
  end_dim = start_dim + ndim

  assert start_dim >= 0 and ndim > 0 and end_dim < array.ndim

  a_shape = array.shape
  i_shape = index.shape

  try:
    common_kernel = torch.broadcast_shapes(
      a_shape[start_dim:end_dim], i_shape[0:ndim]
    )
  except RuntimeError:
    raise ValueError(
      f"vmap dim inconsistent between array and index: {a_shape[start_dim:end_dim]}, {i_shape[0:ndim]}!"
    ) from None

  array = torch.broadcast_to(
    array, a_shape[:start_dim] + common_kernel + a_shape[end_dim:]
  )
  array = torch.reshape(
    array, a_shape[:start_dim] + (-1,) + a_shape[end_dim:]
  )  # May cause a copy

  index = torch.broadcast_to(index, common_kernel + i_shape[ndim:])
  index = torch.reshape(index, (-1,) + i_shape[ndim:])  # May cause a copy

  # TODO mask-less `gather`
  mask = torch.arange(array.shape[start_dim]).reshape(
    (-1,) + tuple(1 for _ in range(index.ndim - 1))
  )

  match start_dim:
    case 0:
      out = array[mask, index]
    case 1:
      out = array[:, mask, index]
    case 2:
      out = array[:, :, mask, index]
    case 3:
      out = array[:, :, :, mask, index]
    case 4:
      out = array[:, :, :, :, mask, index]
    case _:
      raise NotImplementedError(f"Implement more dim plz: {start_dim}")

  return torch.reshape(
    out, out.shape[:start_dim] + common_kernel + out.shape[start_dim + 1:]
  )


def collapse(tensor: torch.Tensor, start_dim: int = -2, end_dim: int | None = None):
  """
  Reshapes `tensor` of shape (d_0, d_1, ..., d_i, d_(i+1), ..., d_j, d_(j+1), ..., d_n)
    to tensor of shape (d_0, d_1, ..., d_(i-1), -1, d_j, d_(j+1), ..., d_n),

  Args:
    tensor - the tensor to reshape
    start_dim - the dimension `i`
    end_dim - the dimension `j`

  Returns:
    reshaped tensor
  """
  original_shape = tensor.shape
  if end_dim is None:
    end_dim = len(original_shape)
  new_shape = original_shape[:start_dim] + (-1,) + original_shape[end_dim:]
  return tensor.reshape(new_shape)


@contextlib.contextmanager
def fp32_matmul_precision(precision: str):
  old_precision = DEFAULT_FP32_MATMUL_PRECISION
  try:
    torch.set_float32_matmul_precision(precision)
    yield
  except:
    torch.set_float32_matmul_precision(old_precision)


def format_time(secs: float | None) -> str:
    if secs is None:
        return "NA"
    ms, s = math.modf(secs)
    ms *= 1000
    min_, s = divmod(s, 60)
    hr, min_ = divmod(min_, 60)
    s = int(s)
    min_ = int(min_)
    hr = int(hr)
    if hr > 0:
        return f"{hr}h {min_}m {s}.{ms:03.0f} s"
    if min_ > 0:
        return f"{min_}m {s}.{ms:03.0f} s"
    if s > 0:
        return f"{s}.{ms:03.0f} s"
    if ms > 10:
        return f"{ms:.0f} ms"
    if ms > 1:
        return f"{ms:.3f} ms"
    us = ms * 1000
    if us > 10:
        return f"{us:.0f} μs"
    if us > 1:
        return f"{us:.3f} μs"
    ns = us * 1000
    return f"{ns:.0f} ns"

def check_tensor_sanity(x: torch.Tensor, name: str | None = None):
  if (
      not DEBUG
      or torch.compiler.is_compiling()
      or compiler.is_precompiling()
  ):
    return
  has_nan = x.isnan().any()
  msg = None
  if has_nan:
    msg = "NaN detected"
  has_inf = x.isinf().any()
  if has_inf:
    msg = "Inf detected"
  has_large_value = (x.abs() > 1e7).any()
  if has_large_value:
    msg = "Large value detected"
  if msg:
    if name:
      msg += f" on `{name}`"
    raise ValueError(msg)

def manual_seed(seed: int):
  """
  Sets all random seed needed for model inference
  """
  from sparsefold import backends as acceleration
  
  torch.manual_seed(seed)
  acceleration.manual_seed(seed)

def func_hash(func: FunctionType):
  r"""
  Return a fingerprint identifying a function's logic.
  The fingerprint uses the function name, code object, and captured closure values.
  """
  if not isinstance(func, FunctionType):
    raise TypeError("Input must be a function object.")

  func_name = func.__name__

  code = func.__code__
  code_hash = hashlib.sha256(
    code.co_code  # Bytecode
    + b"".join([str(c).encode() for c in code.co_consts])  # Constant
    + b"".join([str(name).encode() for name in code.co_names])  # Vars
  ).hexdigest()

  # Get hash of captured vars
  closure_hash = ""
  if func.__closure__:
    closure_vars = []
    for cell in func.__closure__:
      if cell.cell_contents is not None:
        closure_vars.append(cell.cell_contents)
    closure_hash = hashlib.sha256(
      b"".join([str(var).encode() for var in closure_vars])
    ).hexdigest()

  fingerprint = hashlib.sha256(
    (func_name + code_hash + closure_hash).encode()
  ).hexdigest()

  return fingerprint


def _get_caller_info(offset: int = 0):
  """
  Return the caller's function name, filename, and line number.

  Returns:
      tuple: (funcname, filename, lineno).
  """
  USE_UNSAFE: Final[bool] = True

  try:
    if USE_UNSAFE:
      frame = sys._getframe(2 + offset)
    else:
      frame = inspect.currentframe()
      frame = frame.f_back.f_back
      while offset > 0:
        frame = frame.f_back
        offset -= 1

    if frame:
      # Get the caller's filename and line number.
      filename = frame.f_code.co_filename
      lineno = frame.f_lineno
      funcname = frame.f_code.co_name
      return funcname, filename, lineno
    else:
      return None, None, None
  except:
    frame = None
  finally:
    del frame
