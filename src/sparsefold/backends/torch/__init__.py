# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Reference Torch backend: no native JIT, parameter packing or ATen redirects."""
from contextlib import nullcontext
import torch


def apply() -> None:
  return None


def manual_seed(seed: int) -> None:
  torch.manual_seed(seed)


def prepare(module) -> dict[str, int]:
  return {"packed": 0, "fallback": 0, "opaque": 0}


def materialize(*, parallelism: int | None = None) -> int:
  return 0


def precompile(module, *args, **kwargs):
  """Torch compiled regions initialize on first execution; no native warm-up."""
  return None


def is_precompiling() -> bool:
  return False


def execution_context():
  return nullcontext()
