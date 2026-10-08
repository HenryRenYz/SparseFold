# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

from typing import Any, Mapping
from functools import partial
import os

import torch

from sparsefold.runtime import config


def is_precompiling() -> bool:
  # Keep model code independent of the selected acceleration implementation.
  # Import lazily to avoid a compiler <-> acceleration import cycle.
  from sparsefold import backends as acceleration
  return acceleration.is_precompiling()

def is_enabled() -> bool:
  return config.enable_compile

def precompile(
    model,
    *args,
    use_real_tensors: bool = False,
    parallelism: int | None = None,
    capture_graph: bool = False,
    graph_name: str | None = None,
    **kwargs,
):
  """Precompile through the active acceleration backend."""
  from sparsefold import backends as acceleration

  return acceleration.precompile(
      model,
      *args,
      use_real_tensors=use_real_tensors,
      parallelism=parallelism,
      capture_graph=capture_graph,
      graph_name=graph_name,
      **kwargs,
  )

_torch_inductor_options: Mapping[str, Any] = {
    # CACHING
    "fx_graph_cache": True,
    "autotune_local_cache": True, 
    # "fx_graph_remote_cache": True,  
    # "autotune_remote_cache": True, 
    # "bundled_autotune_remote_cache": True,  
    
    # OPTIMIZATION
    "can_inplace_pad_graph_input": True,
    "cpp_wrapper": True,
    "nan_asserts": True,
    "online_softmax": True,
    "dce": True,
    "use_fast_math": True,
    # "max_autotune": True,
    "layout_optimization": True,
    "loop_ordering_after_fusion": True,
    #> TODO requires testing
    "permute_fusion": True,
    # "freezing": True,
    "epilogue_fusion": True,
    "prologue_fusion": True,
    "aggressive_fusion": True,
    "memory_planning": True,
    "memory_pool": "combined",
    "b2b_gemm_pass": True,
    "compile_threads": 16,
    "keep_output_stride": False,
    "enable_linear_binary_folding": True,
    "cpp.max_horizontal_fusion_size": 64,
    "cpp.enable_unsafe_math_opt_flag": True,
    #> Note: might cause numerical issue
    #> off, on, fast
    # "cpp.enable_floating_point_contract_flag": "fast", 
    
    # DEBUG
    # "aot_inductor.debug_compile": True, 
    # "aot_inductor.compile_wrapper_opt_level": 0,
    #> option for debug printing/saving for intermediate tensor values for aot inductor
    #> 0: disable debug dumping
    #> 1: enable saving intermediate tensor values
    #> 2: enable printing intermediate tensor values
    # !3: enable printing kernel names only (useful for pinpointing troublesome kernels)
    # "aot_inductor.debug_intermediate_value_printer": 0,
    # "force_same_precision": True,
    # "emulate_precision_casts": True,
}

# Torch 2.14 aarch64's cpp-wrapper kernels have produced runtime bounds
# failures (index out of bounds in generated .main.cpp); allow falling
# back to the Python wrapper codegen for diagnosis.
if os.environ.get("SPARSEFOLD_INDUCTOR_CPP_WRAPPER", os.environ.get("AF3_INDUCTOR_CPP_WRAPPER")) == "0":
  _torch_inductor_options["cpp_wrapper"] = False

def mark_compile(model=None, /, *, fullgraph=True, dynamic=False, region=None):
  if model is None:
    return partial(
        mark_compile, fullgraph=fullgraph, dynamic=dynamic, region=region
    )
  
  compiled = model
  if config.enable_compile:
    compiled = torch.compile(
        model,
        fullgraph=fullgraph,
        dynamic=dynamic,
        options=_torch_inductor_options,
    )
    # Torch >= 2.14 refuses to compile frames running under a non-infra
    # TorchDispatchMode; the vecops graph-capture wrapper is exactly that,
    # so compiled regions skip it (graph capture for the memory planner
    # is unavailable under compile on those versions).
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 14):
      return compiled
  if not config.uses_vecops():
    return compiled
  try:
    from vecops.graph import wrap_region
  except ImportError:
    return compiled
  return wrap_region(compiled, name=region)
