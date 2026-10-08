# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import contextlib
from collections.abc import Mapping

import torch

Param = Mapping[str, Mapping[str, torch.Tensor]]

class ParamVisitor:
  def __init__(self, params: Param, jax_linear: bool = True):
    '''

      jax_linear: JAX's weight shapes (in_dim, out_dim)
        where PyTorch's weight shapes (out_dim, in_dim)
        that is the transpose of the other.
    '''
    self.params = params
    self._prefix = ''
    self._marks = []
    self._slice: list[int] = []
    self.jax_linear = jax_linear

  def reset(self):
    self._prefix = ''
    self._marks = []
    self._slice = []

  @contextlib.contextmanager
  def into(self, module_name: str, sep: str = '/'):
    if module_name is None:
      raise ValueError(f'module_name is None (when visiting {self.prefix})')
    try:
      self.enter(module_name, sep=sep)
      yield self.prefix
    finally:
      self.leave()

  @contextlib.contextmanager
  def slice(self, index: int):
    try:
      self._slice.append(index)
      yield
    finally:
      self._slice.pop()

  def set(self, module: torch.nn.Module, *, name: str = None, alt_name: str = None, sep: str = '/', **kwargs):
    from sparsefold.model.layers import primitives as fnn
    if isinstance(module, fnn.Module):
      real_name = name
      if real_name is None:
        if hasattr(module, 'name'):
          real_name = module.name
        else:
          real_name = alt_name
      if real_name is None or real_name == '<none>':
        real_name = ''
        sep = ''
      with self.into(real_name, sep=sep):
        module.set_parameters(self, **kwargs)
    elif isinstance(module, (fnn.ModuleList, fnn.ModuleDict)):
      module.set_parameters(self, sep=sep, **kwargs)
    else:
      pass

  def enter(self, module_name: str, *, sep: str = '/'):
    self._marks.append(len(self._prefix))
    if len(self._prefix) == 0:
      self._prefix = module_name
    else:
      self._prefix = f'{self._prefix}{sep}{module_name}'

  def leave(self):
    if len(self._marks) == 0:
      return
    idx = self._marks.pop()
    self._prefix = self._prefix[0:idx]

  @property
  def prefix(self):
    return self._prefix

  def __getitem__(self, name: str) -> torch.Tensor:
    obj = self.params[self._prefix][name]
    if not isinstance(obj, torch.Tensor):
      raise ValueError(f'Parameters is not a PyTorch Tensor: {type(obj)}')
    return obj[tuple(self._slice)]

  def __contains__(self, name: str) -> bool:
    return self._prefix in self.params and name in self.params[self._prefix]
