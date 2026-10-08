# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import torch
import time
import os
import resource
import functools
import contextlib
import fnmatch
import ml_dtypes
import numpy as np
from typing import Callable

from sparsefold.runtime import config
from sparsefold.runtime import util


ENABLE_TRACE: bool = True
LOG_IN_COMPILE_REGION: bool = "+compiled" in config.logs  # still print inside compile region
LOG_TIME: bool = "time" in config.logs
LOG_INFO: bool = "info" in config.logs or "debug" in config.logs
LOG_DEBUG: bool = "debug" in config.logs
# match pattern (file path, i.e. evoformer/pairformer/0) -> logged slice, 
# `()`` means full tensor, this can be huge
LOG_TENSORS: dict[str, util.Slice] = {}
# LOG_TENSORS: dict[str, util.Slice] = {
#     "*/probs": (0, 0),
#     # "*": (),
# }

# type: list[str]
_current_path: list[str] = []
_time_start: dict[str, tuple[float, float, resource.struct_rusage]] = {}


def _log_all_ranks() -> bool:
    return os.environ.get("SPARSEFOLD_TRACE_ALL_RANKS", os.environ.get("AF3_TRACE_ALL_RANKS")) == "1"


def current_path(sep='/') -> str:
    return sep.join(_current_path)

def _should_skip_log() -> bool:
    return not LOG_IN_COMPILE_REGION and torch.compiler.is_compiling() 

def _should_skip_trace() -> bool:
    return not ENABLE_TRACE or _should_skip_log()

@contextlib.contextmanager
def trace(enable: bool=True):
    global ENABLE_TRACE
    old_enable = ENABLE_TRACE
    try:
        ENABLE_TRACE = enable
        yield
    finally:
        ENABLE_TRACE = old_enable

def enter(name, log=True):
    if _should_skip_trace():
        return
    @torch.compiler.disable
    def run():
        _current_path.append(str(name))
        try:
            if not config.uses_vecops():
                raise ImportError
            from vecops.graph import enter_scope

            enter_scope(str(name), kind="sparsefold_scope")
        except ImportError:
            pass
        if log and LOG_TIME:
            key = current_path()
            from sparsefold.runtime import dist
            if dist.is_primary() or _log_all_ranks():
                rank = f"rank={dist.get_rank()} " if _log_all_ranks() else ""
                print(f"[{rank}{key}] Running...", end="\r")
            _time_start[key] = (
                time.perf_counter(),
                time.process_time(),
                resource.getrusage(resource.RUSAGE_SELF),
            )
    run()

def leave(log=True):
    if _should_skip_trace():
        return
    @torch.compiler.disable
    def run():
        if log and LOG_TIME:
            key = current_path()
            started = _time_start.get(key)
            if started is not None:
                wall_start, cpu_start, usage_start = started
                elapsed = time.perf_counter() - wall_start
                from sparsefold.runtime import dist
                if dist.is_primary() or _log_all_ranks():
                    rank = f"rank={dist.get_rank()} " if _log_all_ranks() else ""
                    if _log_all_ranks():
                        usage_end = resource.getrusage(resource.RUSAGE_SELF)
                        cpu_ms = (time.process_time() - cpu_start) * 1000
                        print(
                            f"[{rank}{key}] Took {util.format_time(elapsed)} "
                            f"cpu={cpu_ms:.1f}ms "
                            f"nvcsw={usage_end.ru_nvcsw - usage_start.ru_nvcsw} "
                            f"nivcsw={usage_end.ru_nivcsw - usage_start.ru_nivcsw}"
                        )
                    else:
                        print(f"[{rank}{key}] Took {util.format_time(elapsed)}")
                del _time_start[key]
        try:
            if not config.uses_vecops():
                raise ImportError
            from vecops.graph import leave_scope

            leave_scope(kind="sparsefold_scope")
        except ImportError:
            pass
        _current_path.pop()
    run()

def mark(fn=None, /, *, name):
    if fn is None:
        return functools.partial(mark, name=name)
    
    def wrapper(*args, **kwargs):
        try:
            enter(name)
            return fn(*args, **kwargs)
        finally:
            leave()
    
    return wrapper

def into(name, log: bool=True):
    @contextlib.contextmanager
    def empty_fn():
        yield
        
    if _should_skip_trace():
        return empty_fn()
    
    if isinstance(name, torch.Tensor):
        # only support shape () & (1,)
        name = int(name)

    @contextlib.contextmanager
    def inner_fn():
        try:
            enter(name, log=log)
            yield
        finally:
            leave(log=log)
    return inner_fn()

def _map_torch_tensor(x):
    if x.dtype == ml_dtypes.bfloat16:
        x = np.asarray(x).astype(np.float32)
        x = torch.tensor(x).bfloat16()
    else:
        x = torch.tensor(x)
    return x.cpu().clone()

@torch.compiler.disable
def save_tensor(name: str, x: np.ndarray | torch.Tensor | dict[str, np.ndarray | torch.Tensor]):
    output_base = config.trace_output_path
    os.makedirs(output_base, exist_ok=True)
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = _map_torch_tensor(v)
        x = new_x
    else:
        x = _map_torch_tensor(x)
    path = f"{output_base}/{name}.pt"
    i = 1
    while os.path.exists(path):
        path = f"{output_base}/{name}.{i}.pt"
        i += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(x, path)

def _convert_path_name(path):
    return path.replace('/', '-')

# kwargs: dict[str, torch.Tensor | Callable[[util.Slice], torch.Tensor]]
def log_tensor(**kwargs):
    if _should_skip_log():
        return
    
    for name, data in kwargs.items():
        try:
            enter(name, log=False)
            path = current_path()
            
            matched = False
            cut = None
            for patt, cut_ in LOG_TENSORS.items():
                if fnmatch.fnmatch(path, patt):
                    matched = True
                    cut = cut_
                    break
            if not matched:
                continue
            if isinstance(data, Callable):
                data = data(cut)
            elif isinstance(data, torch.Tensor):
                data = util.slice_tensor(data, slice=cut)
            else:
                raise AssertionError(f"Unsupported data type: {type(data)}")
            save_tensor(path, data)
        finally:
            leave(log=False)

def _get_header() -> str:
    from sparsefold.runtime import dist
    curr_path = current_path()
    if dist.is_enabled():
        return f"[{dist.get_rank()}/{dist.get_world_size()}:{curr_path}]"
    else:
        return f"[{curr_path}]"

def log_info(*args, **kwargs):
    if _should_skip_log():
        return

    @torch.compiler.disable
    def run():
        if LOG_INFO:
            print(_get_header(), *args, **kwargs)
    run()

def log_debug(*args, **kwargs):
    if _should_skip_log():
        return
    
    @torch.compiler.disable
    def run():
        if LOG_DEBUG:
            print(_get_header(), *args, **kwargs)
    run()
