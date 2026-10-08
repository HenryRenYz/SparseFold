# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Optional metadata-only observations of successful eager native bridge calls."""
from collections import Counter
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import sys
import torch


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


@contextmanager
def observe_native_calls():
    destination = os.environ.get("SPARSEFOLD_NATIVE_AUDIT_DIR")
    if not destination:
        yield
        return
    import vecops
    from sparsefold.backends.vecops.kernels import registry
    originals = {}
    counts = Counter()
    def wrap(name, operation):
        @functools.wraps(operation)
        def call(*args, **kwargs):
            # Capture the phase before delegation; only successful calls count.
            eager = not vecops.is_precompiling()
            cpu = all(tensor.device.type == "cpu" for tensor in _tensors((args, kwargs)))
            result = operation(*args, **kwargs)
            if eager and cpu:
                counts[name] += 1
            return result
        return call
    for name, operation in vars(registry).copy().items():
        if name.startswith("_run_") and callable(operation):
            originals[name] = operation
            setattr(registry, name, wrap(name, operation))
    try:
        yield
    finally:
        error = sys.exc_info()[0]
        for name, operation in originals.items():
            setattr(registry, name, operation)
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "sparsefold.native-execution-observations.v1",
            "pid": os.getpid(),
            "case": os.environ.get("SPARSEFOLD_NATIVE_AUDIT_CASE"),
            "context_status": "failed" if error else "completed",
            "error_type": error.__name__ if error else None,
            "precompile_calls_excluded": True,
            "meta_calls_excluded": True,
            "includes_parameter_preparation": False,
            "observation": "Successful eager calls through registered vecops Torch bridges; no tensor values recorded.",
            "counts": dict(sorted(counts.items())),
        }
        path = root / f"native-execution-{os.getpid()}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
