# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

from typing import Iterable, Tuple, List, Mapping, Any

import torch.fx._pytree as fx_tree
import torch.utils._pytree as torch_tree
from torch.utils._pytree import MappingKey


###########################################################
#                       PYTREE UTIL                       #
###########################################################
map = torch_tree.tree_map
flatten = torch_tree.tree_flatten
unflatten = torch_tree.tree_unflatten
leaves = torch_tree.tree_leaves
TreeSpec = torch_tree.TreeSpec
PyTree = torch_tree.PyTree


def dataclass(
    cls=None, /, *,
    mappable_dataclass: bool = True,
    frozen: bool = False,
    return_none_fields: bool = False
):
  def _decorator(cls):
    from dataclasses import dataclass, fields

    orig_cls = cls
    cls = dataclass(cls, frozen=frozen)
    names = [x.name for x in fields(cls)]

    def flatten_fn(obj: Any) -> Tuple[List[Any], torch_tree.Context]:
      flattened = []
      flat_names = []
      none_names = []
      for f in names:
        name, val = f, getattr(obj, f)
        if val is not None or return_none_fields:
          flattened.append(val)
          flat_names.append(name)
        else:
          none_names.append(name)
      return flattened, [flat_names, none_names]

    def flatten_fn_spec(obj: Any, spec: torch_tree.TreeSpec) -> List[Any]:
      flattened = []
      for f in names:
        val = getattr(obj, f)
        if val is not None or return_none_fields:
          flattened.append(val)
      return flattened

    def unflatten_fn(values: Iterable[Any], context: torch_tree.Context) -> Any:
      flat_names, none_names = context
      return cls(**dict(zip(flat_names, values)), **dict.fromkeys(none_names))

    def flatten_fn_with_keys(obj: Any) -> Tuple[List[Any], torch_tree.Context]:
      flattened, (flat_names, none_names) = flatten_fn(obj)  # type: ignore[misc]
      return [(MappingKey(k), v) for k, v in zip(flat_names, flattened)], flat_names

    if mappable_dataclass:
      cls = _mappable_dataclass(cls)

    def as_dict(self):
      out = {}
      for name in names:
        val = getattr(self, name)
        if hasattr(val, 'as_dict'):
          val = val.as_dict()
        out[name] = val
      return out

    @classmethod
    def construct_from_dict(cls, dict):
      kw = {}
      for f in fields(cls):
        if f.name in dict:
          val = dict[f.name]
          if isinstance(f.type, type) and isinstance(val, Mapping) and \
              hasattr(f.type, 'construct_from_dict'):
            val = f.type.construct_from_dict(val)
          kw[f.name] = val
      return cls(**kw)

    if not hasattr(cls, "as_dict"):
      cls.as_dict = as_dict
    if not hasattr(cls, "construct_from_dict"):
      cls.construct_from_dict = construct_from_dict

    torch_tree.register_pytree_node(
      cls,
      serialized_type_name=cls.__qualname__,
      flatten_fn=flatten_fn,
      unflatten_fn=unflatten_fn,
      flatten_with_keys_fn=flatten_fn_with_keys
    )
    fx_tree.register_pytree_flatten_spec(
      cls,
      flatten_fn_spec=flatten_fn_spec
    )
    return cls

  if cls is not None:
    return _decorator(cls)
  else:
    return _decorator


def _mappable_dataclass(cls):
  """Exposes dataclass as ``collections.abc.Mapping`` descendent.

  Allows to traverse dataclasses in methods from `dm-tree` library.

  NOTE: changes dataclasses constructor to dict-type
  (i.e. positional args aren't supported; however can use generators/iterables).

  Args:
    cls: A dataclass to mutate.

  Returns:
    Mutated dataclass implementing ``collections.abc.Mapping`` interface.
  """
  import dataclasses, functools, collections.abc

  if not dataclasses.is_dataclass(cls):
    raise ValueError(f"Expected dataclass, got {cls} (change wrappers order?).")

  # Define methods for compatibility with `collections.abc.Mapping`.
  setattr(cls, "__getitem__", lambda self, x: self.__dict__[x])
  setattr(cls, "__len__", lambda self: len(self.__dict__))
  setattr(cls, "__iter__", lambda self: iter(self.__dict__))
  # Override the default `collections.abc.Mapping` method implementation for
  # cleaner visualization. Without this change x.keys() shows the full repr(x)
  # instead of only the dict_keys present. The same goes for values and items.
  setattr(cls, "keys", lambda self: self.__dict__.keys())
  setattr(cls, "values", lambda self: self.__dict__.values())
  setattr(cls, "items", lambda self: self.__dict__.items())

  # Update constructor.
  orig_init = cls.__init__
  all_fields = set(f.name for f in cls.__dataclass_fields__.values())
  init_fields = [f.name for f in cls.__dataclass_fields__.values() if f.init]

  @functools.wraps(orig_init)
  def new_init(self, *orig_args, **orig_kwargs):
    if (orig_args and orig_kwargs) or len(orig_args) > 1:
      raise ValueError(
        "Mappable dataclass constructor doesn't support positional args."
        "(it has the same constructor as python dict)")
    all_kwargs = dict(*orig_args, **orig_kwargs)
    unknown_kwargs = set(all_kwargs.keys()) - all_fields
    if unknown_kwargs:
      raise ValueError(f"__init__() got unexpected kwargs: {unknown_kwargs}.")

    # Pass only arguments corresponding to fields with `init=True`.
    valid_kwargs = {k: v for k, v in all_kwargs.items() if k in init_fields}
    orig_init(self, **valid_kwargs)

  cls.__init__ = new_init

  # Update base class to derive from Mapping
  dct = dict(cls.__dict__)
  if "__dict__" in dct:
    dct.pop("__dict__")  # Avoid self-references.

  # Remove object from the sequence of base classes. Deriving from both Mapping
  # and object will cause a failure to create a MRO for the updated class
  bases = tuple(b for b in cls.__bases__ if b != object)
  cls = type(cls.__name__, bases + (collections.abc.Mapping,), dct)
  return cls
