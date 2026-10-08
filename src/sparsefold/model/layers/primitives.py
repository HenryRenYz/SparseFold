# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

import enum
import numbers
import torch
import torch.nn.functional as F

from typing import Sequence, Callable, TypeVar, NoReturn, Any, Iterable
from sparsefold.runtime import util
from sparsefold.runtime import tree
from sparsefold.runtime import ops
from sparsefold.runtime import cache
from sparsefold.weights import legacy_scopes as pv


###########################################################
#                     TORCH EXTENSION                     #
###########################################################

class Module(torch.nn.Module):
  r"""
  Implements a default `set_parameters` function
  """

  def __init__(self, *args, **kwargs):
    torch.nn.Module.__init__(self, *args, **kwargs)

  def set_parameters(
      self,
      visitor: pv.ParamVisitor,
      **kwargs
  ):
    r"""
    For each 'torch.nn.Module' object found in attributes:
    * If the object has a 'name' attribute, the name to parameter is set to this attribute.
    * Otherwise the name to parameter is set to the attribute name
    * The object must implements 'set_parameters' function
    """
    for name, child in self._modules.items():
      if child is None: 
        continue
      visitor.set(child, alt_name=name, **kwargs)

  def reset(self):
    self._reset(set())
    
  def _reset(self, visited: set[torch.nn.Module]):
    visited.add(self)
    for attr_name in dir(self):
      try:
        attr = getattr(self, attr_name)
        if isinstance(attr, cache.ValueCache):
          attr.reset()
      except AttributeError:
        continue
    for _, child in self._modules.items():
      if child in visited:
        continue
      Module._reset(child, visited)  # even if child is not an instance of Module



class ModuleList(torch.nn.ModuleList):
  r"""
  Implements a default `set_parameters` function
  """

  def __init__(self, *args, **kwargs):
    torch.nn.ModuleList.__init__(self, *args, **kwargs)

  def set_parameters(
      self,
      visitor: pv.ParamVisitor,
      sep: str = '/',
      **kwargs
  ):
    for sub in self:
      real_name = sub.name
      if real_name == '<none>':
        real_name = ''
        sep = ''
      with visitor.into(real_name, sep=sep):
        sub.set_parameters(visitor, **kwargs)


class ModuleDict(torch.nn.ModuleDict):
  r"""
  Implements a default `set_parameters` function
  """

  def __init__(self, *args, **kwargs):
    torch.nn.ModuleDict.__init__(self, *args, **kwargs)

  def set_parameters(
      self,
      visitor: pv.ParamVisitor,
      sep: str = '/',
      **kwargs
  ):
    for sub_name, sub in self.items():
      real_name = sub_name
      if hasattr(sub, 'name'):
        real_name = sub.name
      if real_name == '<none>':
        real_name = ''
        sep = ''
      with visitor.into(real_name, sep=sep):
        sub.set_parameters(visitor, **kwargs)


class LinearPostOp(enum.IntEnum):
  """Post operations supported by :class:`Linear` and its native backend."""

  NONE = 0
  RELU = 1
  SILU = 2
  SIGMOID = 3


def normalize_linear_post_op(
    post_op: str | LinearPostOp | Callable[[torch.Tensor], torch.Tensor] | None,
) -> LinearPostOp:
  """Normalizes the public Linear post-op spelling to a stable enum value."""
  if post_op is None:
    return LinearPostOp.NONE
  if isinstance(post_op, LinearPostOp):
    return post_op
  if isinstance(post_op, str):
    try:
      return LinearPostOp[post_op.strip().upper()]
    except KeyError as exc:
      raise ValueError(f"Unsupported Linear post_op: {post_op!r}") from exc
  if post_op in (torch.relu, F.relu):
    return LinearPostOp.RELU
  if post_op in (torch.sigmoid, F.sigmoid):
    return LinearPostOp.SIGMOID
  if post_op in (F.silu, torch.nn.functional.silu):
    return LinearPostOp.SILU
  raise ValueError(f"Unsupported Linear post_op callable: {post_op!r}")


def apply_linear_post_op(
    output: torch.Tensor,
    post_op: str | LinearPostOp | Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
  """Applies a normalized Linear post operation in the Torch fallback."""
  normalized = normalize_linear_post_op(post_op)
  if normalized == LinearPostOp.NONE:
    return output
  if normalized == LinearPostOp.RELU:
    return torch.relu(output)
  if normalized == LinearPostOp.SILU:
    return F.silu(output)
  if normalized == LinearPostOp.SIGMOID:
    if output.device.type == "cpu":
      return _cpu_bounded_sigmoid(output)
    return torch.sigmoid(output)
  raise AssertionError(f"Unhandled Linear post_op: {normalized}")


def _cpu_bounded_sigmoid(x: torch.Tensor) -> torch.Tensor:
  """CPU sigmoid without Torch BF16's data-dependent SLEEF exp slow path."""
  bounded = x.to(torch.float32).clamp(-20, 20)
  z = -bounded.abs() / 256.0
  exp_value = 1.0 + z * (
      1.0 + z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z / 120.0)))
  )
  for _ in range(8):
    exp_value = exp_value * exp_value
  denominator = 1.0 + exp_value
  return torch.where(
      bounded >= 0, 1.0 / denominator, exp_value / denominator
  ).to(x.dtype)


class Linear(Module):
  """Custom Linear Module.

  This differs from the standard Linear in a few ways:
    * It supports inputs of arbitrary rank
    * It allows to use ntk parametrization
    * Initializers are specified by strings
    * It allows to explicitly specify which dimension of the input will map to
      the tpu sublane/lane dimensions.
    * The weight matrix is the transpose of Torch's Linear
  """

  weight: torch.nn.Parameter

  def __init__(
      self,
      num_input: int | Sequence[int],
      num_output: int | Sequence[int],
      *,
      bias: bool = False,
      fast_scalar_mode: bool = True,
      transpose_weights: bool = False,
      device=None,
      dtype: torch.dtype = None,
      precision: str | None = None,
      name: str = None
  ):
    """Constructs Linear Module.

    ASC Note:
      transpose_weights:
        O = x @ W ^ T, which einsum is '...ij,...kj->...ik'
      otherwise:
        O = x @ W, which einsum is '...ij,...jk->...ik'

    Args:
      num_output: number of output channels. Can be tuple when outputting
        multiple dimensions.
      initializer: What initializer to use, should be one of {'linear', 'relu',
        'zeros'}.
      num_input_dims: Number of dimensions from the end to project.
      use_bias: Whether to include trainable bias (False by default).
      bias_init: Value used to initialize bias.
      precision: What precision to use for matrix multiplication, defaults to
        None.
      fast_scalar_mode: Whether to use optimized path for num_input_dims = 0.
      transpose_weights: decides whether weights have shape [input, output] or
        [output, input], True means [output, input], this is helpful to avoid
        padding on the tensors holding the weights.
      name: name of module, used for name scopes.
    """
    super().__init__()
    self.name = name
    if isinstance(num_output, numbers.Integral):
      self.output_shape = (num_output,)
    else:
      self.output_shape = tuple(num_output)
    if isinstance(num_input, numbers.Integral):
      self.input_shape = (num_input,)
    else:
      self.input_shape = tuple(num_input)
    self.use_bias = bias
    self.num_input_dims = len(self.input_shape)
    self.num_output_dims = len(self.output_shape)
    self.transpose_weights = transpose_weights
    self.slice_order = ()  # no slicing dim
    self.precision = precision
    self.dtype = dtype

    assert self.num_input_dims >= 0

    # self.simple_linear = simple_linear = self.num_input_dims == self.num_output_dims and self.num_input_dims == 1

    if transpose_weights:
      weight_shape = self.output_shape + self.input_shape
    else:
      weight_shape = self.input_shape + self.output_shape
    self.weight_shape = weight_shape

    if self.num_input_dims > 0:
      in_letters = 'abcde'[: self.num_input_dims]
      out_letters = 'hijkl'[: self.num_output_dims]
      if transpose_weights:
        equation = (
          f'...{in_letters}, {out_letters}{in_letters}->...{out_letters}'
        )
      else:
        equation = (
          f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'
        )
      self.equation = equation
    else:
      self.equation = ''  # Placeholder

    self.weight = torch.nn.Parameter(
      torch.empty(weight_shape, dtype=dtype, device=util.DEFAULT_DEVICE if device is None else device)
    )
    if bias:
      self.bias = torch.nn.Parameter(
        torch.empty(self.output_shape, dtype=dtype, device=util.DEFAULT_DEVICE if device is None else device)
      )
    else:
      self.register_parameter("bias", None)

  def set_parameters(
      self,
      visitor: pv.ParamVisitor,
      **kwargs
  ):
    if 'weights' in visitor:
      weight = visitor['weights']
    elif 'weight' in visitor:
      weight = visitor['weight']
    else:
      _checked_raise(f'No weight found in parameters! (prefix={visitor.prefix})')
    if self.dtype is not None:
      weight = weight.to(self.dtype)
    # Actual shape: (slice_order, weight_shape)
    self.weight = torch.nn.Parameter(weight)
    # Check shape
    tailing_shape = self.weight_shape
    actual_shape = weight.shape[-len(tailing_shape):]
    if actual_shape != tailing_shape:
      _checked_raise(
        f'Linear weight shape mismatch: expect {tailing_shape}, got {actual_shape} (prefix={visitor.prefix})')

    weight_slice_order = weight.shape[:-len(tailing_shape)]

    if self.use_bias:
      if 'bias' in visitor:
        bias = visitor['bias']
      else:
        _checked_raise(f'No bias found in parameters! (prefix={visitor.prefix})')
      if self.dtype is not None:
        bias = bias.to(self.dtype)
      # Actual shape: (slice_order, output_shape)
      self.bias = torch.nn.Parameter(bias)
      # Check shape
      tailing_shape = self.output_shape
      actual_shape = bias.shape[-len(tailing_shape):]
      if actual_shape != tailing_shape:
        _checked_raise(
          f'Linear bias shape mismatch: expect {tailing_shape}, got {actual_shape} (prefix={visitor.prefix})')
      bias_slice_order = bias.shape[:-len(tailing_shape)]
      if weight_slice_order != bias_slice_order:
        _checked_raise(
          f'Linear bias slice order mismatch: expect {weight_slice_order}, got {bias_slice_order} (prefix={visitor.prefix})')

    self.slice_order = weight_slice_order

  def forward(
      self,
      inputs: torch.Tensor,
      *,
      post_op: str | LinearPostOp | Callable[[torch.Tensor], torch.Tensor] | None = None,
      residual: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Connects Module.

    Args:
      inputs: Tensor of shape [..., input_shape].
      post_op: Optional fused activation (``relu``, ``silu`` or ``sigmoid``).
      residual: Optional tensor added after the projection and bias, before
        ``post_op``. Broadcasting follows normal Torch semantics.

    Returns:
      output of shape [..., output_shape]
    """

    input_dtype = inputs.dtype
    weight = self.weight
    bias = None
    if self.use_bias:
      bias = self.bias
      assert weight.dtype == bias.dtype
    if input_dtype != weight.dtype:
      inputs = inputs.to(weight.dtype)

    if self.num_input_dims == 0:
      for i in range(-1, -self.num_output_dims - 1, -1):
        inputs = inputs.unsqueeze(dim=i)
      output = inputs * weight
    else:
      if self.precision:
        old_precision = util.DEFAULT_FP32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(self.precision)
      output = torch.einsum(self.equation, inputs, weight)
      if self.precision:
        torch.set_float32_matmul_precision(old_precision)
      if self.use_bias:
        output += bias
    if residual is not None:
      output = output + residual.to(output.dtype)
    output = apply_linear_post_op(output, post_op)
    return output.to(input_dtype)


class LayerNorm(Module):
  r"""
  Special LayerNorm with auto param dtype conversion
  """

  weight: torch.nn.Parameter

  def __init__(
      self,
      normalized_shape: int,
      *,
      eps: float = 1e-5,
      elementwise_affine: bool = True,
      dim: int = -1,
      bias: bool = True,
      device=None,
      dtype=None,
      name: str = None,
  ) -> None:
    super().__init__()
    self.name = name
    self.dim = dim
    self.use_bias = bias
    self.dtype = dtype
    self.slice_order = ()

    if isinstance(normalized_shape, numbers.Integral):
      # mypy error: incompatible types in assignment
      normalized_shape = (normalized_shape,)  # type: ignore[assignment]
    else:
      assert len(normalized_shape) == 1, 'We currently only support 1D layer norm.'
    self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
      self.weight = torch.nn.Parameter(
        torch.empty(self.normalized_shape, dtype=dtype, device=util.DEFAULT_DEVICE if device is None else device)
      )
      if bias:
        self.bias = torch.nn.Parameter(
          torch.empty(self.normalized_shape, dtype=dtype, device=util.DEFAULT_DEVICE if device is None else device)
        )
      else:
        self.register_parameter("bias", None)
    else:
      self.register_parameter("weight", None)
      self.register_parameter("bias", None)

  def set_parameters(
      self,
      visitor: pv.ParamVisitor,
      **kwargs
  ):
    if not self.elementwise_affine:
      return
    if 'weights' in visitor:
      weight = visitor['weights']
    elif 'weight' in visitor:
      weight = visitor['weight']
    elif 'scale' in visitor:
      weight = visitor['scale']
    else:
      _checked_raise(f'No weight found in parameters! (prefix={visitor.prefix})')
    if self.dtype is not None:
      weight = weight.to(self.dtype)
    self.weight = torch.nn.Parameter(weight)
    # Check shape
    actual_shape = weight.shape[-len(self.normalized_shape):]
    if actual_shape != self.normalized_shape:
      _checked_raise(
        f'LayerNorm weight shape mismatch: expect {self.normalized_shape}, got {actual_shape} (prefix={visitor.prefix})')

    weight_slice_order = weight.shape[:-len(self.normalized_shape)]

    bias = None
    # if self.use_bias:
    if 'bias' in visitor:
      bias = visitor['bias']
    elif 'offset' in visitor:
      bias = visitor['offset']
    if bias is not None:
      if self.dtype is not None:
        bias = bias.to(self.dtype)
      self.bias = torch.nn.Parameter(bias)
      # Check shape
      actual_shape = bias.shape[-len(self.normalized_shape):]
      if actual_shape != self.normalized_shape:
        _checked_raise(
          f'LayerNorm weight shape mismatch: expect {self.normalized_shape}, got {actual_shape} (prefix={visitor.prefix})')
      bias_slice_order = bias.shape[:-len(self.normalized_shape)]
      if weight_slice_order != bias_slice_order:
        _checked_raise(
          f'LayerNorm bias slice order mismatch: expect {weight_slice_order}, got {bias_slice_order} (prefix={visitor.prefix})')

    self.slice_order = weight_slice_order

  def forward(
      self,
      input: torch.Tensor,
  ) -> torch.Tensor:
    dim = self.dim
    ndim = len(input.shape)
    if dim < 0:
      dim = ndim + dim
    assert 0 <= dim and dim < ndim
    if dim != ndim - 1:
      input = torch.transpose(input, dim, -1)

    weight = self.weight
    bias = self.bias

    dtype = input.dtype
    input = input.to(self.dtype)
    if weight is not None and input.dtype != weight.dtype:
      weight = weight.to(input.dtype)
    if bias is not None and input.dtype != bias.dtype:
      bias = bias.to(input.dtype)
    # ASC: original precision fp32
    output = F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
    output = output.to(dtype)
    if dim != ndim - 1:
      output = torch.transpose(output, dim, -1)
    return output


class LayerStack(Module, torch.nn.ModuleList):
  r"""
  Helper class to add prefix to the parameter path (otherwise no difference to ModuleList)
  """

  def __init__(
      self,
      blocks: Iterable[torch.nn.Module],
      *,
      with_per_layer_inputs: bool = False,
      name: str = None
  ):
    Module.__init__(self)
    torch.nn.ModuleList.__init__(self, blocks)
    if with_per_layer_inputs:
      prefix = '__layer_stack_with_per_layer'
    else:
      prefix = '__layer_stack_no_per_layer'
    if name is None:
      name = prefix
    else:
      name = prefix + name
    self.name = name
  
  def set_parameters(self, visitor, **kwargs):
    for i, block in enumerate(self):
      with visitor.slice(i):
        visitor.set(block, **kwargs)

  def forward(self, *args, **kwargs):
    raise AssertionError("Do not call LayerStack.forward directly, use layer_stack!")


_Carry = TypeVar('_Carry')

layer_stack_indices: list[int] = []

def layer_stack(
    block: Callable[[torch.nn.Module, _Carry], _Carry] | Callable[[torch.nn.Module, _Carry, Any], _Carry],
    modules: tree.PyTree = (),
    /, *,
    num_layers: int | None = None,
    with_per_layer_inputs: bool = False,
    unroll: int | bool | None = None,
    use_hop: bool = False,
) -> Callable[[_Carry], _Carry]:
  flattened_modules = tree.leaves(modules)
  assert len(flattened_modules) > 0 or num_layers is not None, "Neither modules nor num_layers is specified"
  if len(flattened_modules) > 0:
    assert all(isinstance(x, torch.nn.ModuleList) for x in flattened_modules), "All modules should be object of torch.nn.ModuleList"
    deduced_num_layers = len(flattened_modules[0])
    if num_layers is not None:
      assert deduced_num_layers == num_layers, "num_layers must be consistent with actual length of modules"
    else:
      assert all(len(x) == deduced_num_layers for x in flattened_modules[1:]), "All modules should have the same length"
      num_layers = deduced_num_layers
  
  if with_per_layer_inputs:
    def func(init, input):
      assert len(input) == num_layers

      def wrap(i, carry, input):
        layer_stack_indices[-1] = i
        local_modules = tree.map(lambda x: x[i], modules)
        return block(local_modules, carry, input)

      return ops.indexed_stack_scan(wrap, init, input, length=num_layers, unroll=unroll, use_hop=use_hop)
  else:
    def func(init):
      def wrap(i, carry):
        layer_stack_indices[-1] = i
        local_modules = tree.map(lambda x: x[i], modules)
        return block(local_modules, carry)

      return ops.indexed_stack_scan(wrap, init, length=num_layers, unroll=unroll, use_hop=use_hop)
    
  def func_with_counter_guard(*args):
    try:
      layer_stack_indices.append(0)
      return func(*args)
    finally:
      layer_stack_indices.pop()

  return func_with_counter_guard


class GatedLinearUnit(Module):
  def __init__(self, activation: torch.nn.Module | None = None):
    super().__init__()
    self.activation = activation

  def forward(
      self,
      x: torch.Tensor,  # (*B M K)
      weight: torch.Tensor  # (K 2 N)
  ) -> torch.Tensor:  # (*B M N)
    """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

    Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`.

    This is SwiGLU when `activation=jax.nn.swish`, GEGLU when
    `activation=jax.nn.gelu`, REGLU when `activation=jax.nn.relu`, and GLU when
    `activation=jax.nn.sigmoid` (https://arxiv.org/abs/2002.05202).

    Args:
      x: the input array.
      weight: the combined weight array.
      activation: optional activation function.
      precision: specifies the matrix multiplication precision. Either `None`
        (default), which means the default precision for the backend, or a
        `jax.lax.Precision` enum.

    Returns:
      The output array.
    """

    weight_reshaped = util.collapse(
      weight, start_dim=-2, end_dim=None
    )
    assert weight_reshaped.ndim == 2

    if x.dtype != weight_reshaped.dtype:
      weight_reshaped = weight_reshaped.to(x.dtype)

    y = torch.einsum('...ij,...jk->...ik', x, weight_reshaped)

    # Apply activation and compute product of FP8/FP16/BF16 in FP32.
    # y = y.astype(jnp.promote_types(x.dtype, jnp.float32))
    a, b = torch.tensor_split(y, 2, dim=-1)
    out = a * b if self.activation is None else self.activation(a) * b
    out = out.to(x.dtype)
    return out


class Swish(Module):
  r"""
    Swish activation, defined as:
        $ swish(x) = x \cdot \sigma (x) $
    where $\sigma (x)$ is the sigmoid function.
  """

  def forward(self, x: torch.Tensor):
    return swish(x)


def gated_linear_unit(
    x: torch.FloatTensor,  # (*B M K)
    weight: torch.FloatTensor,  # (K 2 N)
    activation: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.FloatTensor:  # (*B M N)
  """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

  Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`.

  This is SwiGLU when `activation=jax.nn.swish`, GEGLU when
  `activation=jax.nn.gelu`, REGLU when `activation=jax.nn.relu`, and GLU when
  `activation=jax.nn.sigmoid` (https://arxiv.org/abs/2002.05202).

  Args:
    x: the input array.
    weight: the combined weight array.
    activation: optional activation function.
    precision: specifies the matrix multiplication precision. Either `None`
      (default), which means the default precision for the backend, or a
      `jax.lax.Precision` enum.

  Returns:
    The output array.
  """

  weight_reshaped = util.collapse(
    weight, start_dim=-2, end_dim=None
  )
  assert weight_reshaped.ndim == 2

  if x.dtype != weight_reshaped.dtype:
    weight_reshaped = weight_reshaped.to(x.dtype)

  y = torch.matmul(x, weight_reshaped)

  # Apply activation and compute product of FP8/FP16/BF16 in FP32.
  # y = y.astype(jnp.promote_types(x.dtype, jnp.float32))
  a, b = torch.tensor_split(y, 2, dim=-1)
  out = a * b if activation is None else activation(a) * b
  out = out.to(x.dtype)
  return out


def swish(x: torch.Tensor):
  r"""
    Swish activation, defined as:
        $ swish(x) = x \cdot \sigma (x) $
    where $\sigma (x)$ is the sigmoid function.
  """
  if x.device.type == "cpu":
    # Keep the SwiGLU fallback off Torch's BF16 sigmoid slow path too. The
    # negative tail is zeroed so a huge negative x cannot amplify 2e-9.
    sigmoid = _cpu_bounded_sigmoid(x)
    return torch.where(x < -20, torch.zeros_like(x), x * sigmoid)
  return x * torch.sigmoid(x)


def one_hot(
    x: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float32,
    value: float = 1.0,
    device: str | torch.device | None = None
) -> torch.Tensor:
  r"""
  x: (...), int
  returns (*x.shape, num_classes), dtype
  """
  if device is None:
    device = x.device
  if isinstance(device, str):
    device = torch.device(device)
  y = torch.zeros(x.shape + (num_classes,), dtype=dtype, device=util.DEFAULT_DEVICE if device is None else device)
  y = y.scatter(dim=-1, index=x.long()[..., None], value=value)
  return y


###########################################################
#                     INTERNAL STUFF                      #
###########################################################

def _checked_raise(msg) -> NoReturn:
  if torch.compiler.is_compiling():
    print('Exception raised during compilation: ' + msg)
  raise ValueError(msg)
