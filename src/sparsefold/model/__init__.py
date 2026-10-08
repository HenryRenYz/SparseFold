# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Numerical model package; public network import is lazy."""
__all__ = ["SparseFoldModel"]

def __getattr__(name):
    if name == "SparseFoldModel":
        from sparsefold.model.network import SparseFoldModel
        return SparseFoldModel
    raise AttributeError(name)
