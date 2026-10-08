# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
"""Audited legacy-scope loading independent of SparseFold attribute names.

The checkpoint namespace is stable. Module ``name`` values describe that namespace;
Python attributes and canonical state keys may change without rewriting source keys.
Only identity storage plus dtype/device conversion is permitted by this loader.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
import threading
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .legacy_scopes import ParamVisitor
from .records import Checkpoint


WEIGHT_MAP_SCHEMA_ID = "sparsefold.weight-map.v1"
SOURCE_FORMAT_ID = "af3-record-stream.v1"
TENSOR_STORAGE_ID = "jax-logical-layout"


@dataclass(frozen=True)
class MappingEntry:
    source_scope: str
    source_name: str
    stack_indices: tuple[int, ...]
    target_key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    transforms: tuple[str, ...]


@dataclass(frozen=True)
class LoadReport:
    entries: tuple[MappingEntry, ...]
    missing_model_keys: tuple[str, ...] = ()
    unexpected_source_keys: tuple[str, ...] = ()
    uncovered_stack_slices: tuple[str, ...] = ()
    identifier: bytes | None = None

    @property
    def complete(self) -> bool:
        return not (self.missing_model_keys or self.unexpected_source_keys or self.uncovered_stack_slices)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible mapping audit (no tensor values).

        ``source_format`` identifies the AF3 binary record convention consumed by
        this adapter; it is a SparseFold format label, not an upstream API promise.
        ``tensor_storage`` preserves scientific dimensions including stacked axes.
        """
        result = asdict(self)
        result.update(schema_id=WEIGHT_MAP_SCHEMA_ID,
                      source_format=SOURCE_FORMAT_ID,
                      tensor_storage=TENSOR_STORAGE_ID,
                      complete=self.complete)
        # Metadata is an arbitrary byte identifier, never assume text encoding.
        result['identifier'] = self.identifier.hex() if self.identifier is not None else None
        return result


class WeightLoadError(ValueError):
    def __init__(self, message: str, report: LoadReport | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class _Access:
    scope: str
    name: str
    indices: tuple[int, ...]
    tensor: torch.Tensor
    source_shape: tuple[int, ...]
    copied_inference_source: bool = False


def _storage_key(tensor: torch.Tensor) -> tuple:
    return (str(tensor.device), tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


class _ProvenanceMode(TorchDispatchMode):
    """Track allowed storage conversions rather than comparing full weight values."""
    def __init__(self):
        super().__init__()
        self.provenance: dict[tuple, _Access] = {}

    def add(self, tensor: torch.Tensor, access: _Access) -> None:
        self.provenance[_storage_key(tensor)] = access

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        # Parameter construction detaches; .to and torch.as_tensor can copy dtype/device.
        allowed = {torch.ops.aten.detach.default, torch.ops.aten.alias.default,
                   torch.ops.aten._to_copy.default, torch.ops.aten.clone.default}
        if func in allowed and args and isinstance(args[0], torch.Tensor):
            access = self.provenance.get(_storage_key(args[0]))
            if access is not None and isinstance(result, torch.Tensor):
                self.add(result, access)
        return result


class _AuditedVisitor(ParamVisitor):
    def __init__(self, params, provenance: _ProvenanceMode):
        super().__init__(params)
        self.provenance = provenance
        self.accesses: list[_Access] = []
        self.probes: list[tuple[str, str, bool]] = []

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            tensor = super().__getitem__(name)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WeightLoadError(f'Missing or invalid checkpoint tensor: {self.prefix}/{name} slice={self._slice}: {exc}') from exc
        copied_inference_source = tensor.is_inference()
        if copied_inference_source:
            # Disabling inference mode does not retroactively make existing
            # inference tensors ordinary. Copy this selected slice before
            # Parameter construction so dtype/device provenance remains visible.
            tensor = tensor.clone()
        access = _Access(self.prefix, name, tuple(self._slice), tensor,
                         tuple(self.params[self.prefix][name].shape), copied_inference_source)
        self.accesses.append(access)
        self.provenance.add(tensor, access)
        return tensor

    def __contains__(self, name: str) -> bool:
        result = super().__contains__(name)
        self.probes.append((self.prefix, name, result))
        return result


_LOAD_LOCK = threading.RLock()


def load_checkpoint(model: torch.nn.Module, checkpoint: Checkpoint | Mapping, *, strict: bool = True) -> LoadReport:
    """Load strictly even when called from an inference-mode execution context.

    Parameter initialization requires visible dtype/device dispatch. Existing
    inference checkpoint tensors are copied per selected slice into ordinary
    tensors; the caller's inference-mode state is restored when this returns.
    """
    with torch.inference_mode(False), torch.no_grad():
        return _load_checkpoint(model, checkpoint, strict=strict)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Checkpoint | Mapping, *, strict: bool = True) -> LoadReport:
    """Assign local legacy tensors and return an explicit scope/slice -> state-key map.

    This is intended for a freshly constructed CPU model with its modules created.
    A failed load restores original Parameter objects. The temporary registration
    hook sees only the supplied model; the process-wide hook is removed in all cases.
    All source slices must be covered in strict mode, including stack leading axes.
    ``strict=False`` only permits unused tensors/parameters; invalid assignments,
    duplicate assignments and transforms still fail.
    """
    if isinstance(checkpoint, Checkpoint):
        params, identifier, raw_scopes = checkpoint.params, checkpoint.identifier, checkpoint.source_scopes
    else:
        params, identifier, raw_scopes = checkpoint, None, {}
        if '__meta__' in params:
            raise WeightLoadError('Use read_checkpoint to separate and validate checkpoint metadata')
    if not isinstance(params, Mapping):
        raise TypeError('Checkpoint params must be a nested mapping')
    for scope, tensors in params.items():
        if not isinstance(scope, str) or not isinstance(tensors, Mapping):
            raise WeightLoadError('Checkpoint scopes must map names to tensors')
        for name, tensor in tensors.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise WeightLoadError(f'Checkpoint value must be a tensor: {scope}/{name}')
    modules = dict(model.named_modules())
    module_paths = {id(module): path for path, module in modules.items()}
    original = {id(module): dict(module._parameters) for module in modules.values()}
    expected = set(dict(model.named_parameters()))
    entries: list[MappingEntry] = []
    targets: set[str] = set()
    target_accesses: dict[str, _Access] = {}
    entry_indices: dict[str, int] = {}
    consumed: set[tuple[str, str, tuple[int, ...]]] = set()
    provenance = _ProvenanceMode()
    visitor = _AuditedVisitor(params, provenance)

    def assign(module, name, parameter):
        if id(module) not in module_paths or parameter is None:
            return
        path = module_paths[id(module)]
        target = f'{path}.{name}' if path else name
        access = provenance.provenance.get(_storage_key(parameter))
        if access is None:
            raise WeightLoadError(f'Unmapped parameter assignment or unsupported transform: {target}')
        source = (access.scope, access.name, access.indices)
        updating = target in targets and target_accesses[target] is access
        if not updating and (target in targets or source in consumed):
            raise WeightLoadError(f'Duplicate checkpoint mapping: {source} -> {target}')
        if tuple(parameter.shape) != tuple(access.tensor.shape):
            raise WeightLoadError(f'Unexpected shape transform: {source} -> {target}')
        transforms = []
        if access.copied_inference_source:
            transforms.append('inference_tensor_copy')
        if access.indices:
            transforms.append('leading_stack_slice')
        if parameter.dtype != access.tensor.dtype:
            transforms.append(f'dtype:{access.tensor.dtype}->{parameter.dtype}')
        if parameter.device != access.tensor.device:
            transforms.append(f'device:{access.tensor.device}->{parameter.device}')
        if tuple(parameter.stride()) != tuple(access.tensor.stride()):
            transforms.append('contiguous')
        if not transforms:
            transforms.append('identity')
        entry = MappingEntry(raw_scopes.get(access.scope, access.scope), access.name,
                             access.indices, target, access.source_shape,
                             tuple(parameter.shape), tuple(transforms))
        if updating:
            entries[entry_indices[target]] = entry
        else:
            entry_indices[target] = len(entries)
            entries.append(entry)
            target_accesses[target] = access
        targets.add(target)
        consumed.add(source)

    with _LOAD_LOCK:
        handle = torch.nn.modules.module.register_module_parameter_registration_hook(assign)
        try:
            with provenance:
                visitor.set(model)
            keys = {(scope, name) for scope, tensors in params.items() for name in tensors}
            used = {(scope, name) for scope, name, _ in consumed}
            missing = tuple(sorted(expected - targets))
            unexpected = tuple(sorted(f'{scope}/{name}' for scope, name in keys - used))
            uncovered = []
            grouped: dict[tuple[str, str], list[tuple[int, ...]]] = defaultdict(list)
            for scope, name, indices in consumed:
                grouped[(scope, name)].append(indices)
            for (scope, name), slices in grouped.items():
                depths = {len(indices) for indices in slices}
                if len(depths) != 1:
                    raise WeightLoadError(f'Inconsistent stack slice depth: {scope}/{name}')
                depth = next(iter(depths))
                count = math.prod(params[scope][name].shape[:depth])
                if len(slices) != count:
                    uncovered.append(f'{scope}/{name}: {len(slices)}/{count} stack slices')
            read_sources = {(access.scope, access.name, access.indices) for access in visitor.accesses}
            if read_sources != consumed:
                raise WeightLoadError('Checkpoint tensors were read without a mapped parameter assignment')
            report = LoadReport(tuple(entries), missing, unexpected, tuple(sorted(uncovered)), identifier)
            if strict and not report.complete:
                raise WeightLoadError(f'Incomplete checkpoint mapping: missing={missing}, unexpected={unexpected}, uncovered={tuple(uncovered)}', report)
            return report
        except Exception as exc:
            for module in modules.values():
                module._parameters.clear()
                module._parameters.update(original[id(module)])
            if isinstance(exc, WeightLoadError):
                raise
            raise WeightLoadError(f'Checkpoint loading failed: {exc}') from exc
        finally:
            handle.remove()
