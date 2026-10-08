# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import torch
from typing import Any, Callable
from sparsefold.runtime import util
from sparsefold.runtime import config


def _get_ident_value(x):
  if isinstance(x, torch.Tensor):
    return id(x)
  else:
    return x

class ValueCache:
  def __init__(self):
    self._cache: dict[tuple[int, ...], dict[Any, Any]] = {}
  
  @torch.compiler.disable
  def get(self, *args, name: str | Any | None = None, factory: Callable[[], Any] | None = None) -> Any | None:
    if not config.enable_cache:
      return factory()
    ids = tuple(_get_ident_value(x) for x in args)
    map_ = self._cache.setdefault(ids, {})
    if name is None:
      funcname, filename, fileno = util._get_caller_info()
      name = f"{funcname}@{filename}:{fileno}"
    value = map_.get(name)
    if value is None and factory is not None:
      map_[name] = value = factory()
    if value is not None and config.uses_vecops():
      try:
        from vecops.graph import mark_retained

        mark_retained(value, reason=f"sparsefold.ValueCache:{name}")
      except ImportError:
        pass
    return value

  def clear(self, *args):
    ids = {_get_ident_value(x) for x in args}
    to_remove = []
    for coll in self._cache.keys():
      intersect = ids.intersection(coll)
      if intersect:
        to_remove.append(coll)
    for coll in to_remove:
      del self._cache[coll]
    
  def reset(self):
    self._cache.clear()
