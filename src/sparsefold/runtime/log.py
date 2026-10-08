# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

from sparsefold.runtime import config

def info(*args, **kwargs):
  from sparsefold.runtime import dist
  if dist.is_enabled():
    print(f"[{dist.get_rank()}/{dist.get_world_size()}]", *args, **kwargs)
  else:
    print(*args, **kwargs)

def warning(*args, **kwargs):
  from sparsefold.runtime import dist
  if dist.is_enabled():
    print(f"[{dist.get_rank()}/{dist.get_world_size()}]", *args, **kwargs)
  else:
    print(*args, **kwargs)

def error(*args, **kwargs):
  from sparsefold.runtime import dist
  if dist.is_enabled():
    print(f"[{dist.get_rank()}/{dist.get_world_size()}]", *args, **kwargs)
  else:
    print(*args, **kwargs)
