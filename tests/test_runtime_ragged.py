# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Tests for ragged result gathering without an AllGatherV dependency."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sparsefold.runtime import dist


def _ragged_shard() -> dist.Shard:
  shard = object.__new__(dist.Shard)
  shard.length = 4
  shard.num_shards = 3
  shard.is_even = False
  shard._sizes = [2, 1, 1]
  shard._offsets = [0, 2, 3]
  return shard


def test_padded_gather_removes_rank_padding(monkeypatch):
  shard = _ragged_shard()
  monkeypatch.setattr(dist, "get_rank", lambda: 1)
  monkeypatch.setattr(dist, "get_world_size", lambda: 3)

  def fake_all_gather(output: torch.Tensor, input: torch.Tensor):
    assert input.tolist() == [[10.0, 11.0], [0.0, 0.0]]
    output.copy_(
        torch.tensor([
            [0.0, 1.0],
            [2.0, 3.0],
            [10.0, 11.0],
            [-1.0, -1.0],
            [20.0, 21.0],
            [-1.0, -1.0],
        ])
    )

  monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather)
  result = shard.all_gather_padded_into_tensor(
      input_tensor=torch.tensor([[10.0, 11.0]])
  )
  assert result.tolist() == [
      [0.0, 1.0],
      [2.0, 3.0],
      [10.0, 11.0],
      [20.0, 21.0],
  ]


def test_padded_gather_supports_nonleading_dimension(monkeypatch):
  shard = _ragged_shard()
  monkeypatch.setattr(dist, "get_rank", lambda: 1)
  monkeypatch.setattr(dist, "get_world_size", lambda: 3)

  def fake_all_gather(output: torch.Tensor, input: torch.Tensor):
    assert input.shape == (2, 2)
    output.copy_(
        torch.tensor([
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [-1.0, -1.0],
            [3.0, 13.0],
            [-1.0, -1.0],
        ])
    )

  monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather)
  result = shard.all_gather_padded_into_tensor(
      input_tensor=torch.tensor([[2.0], [12.0]]), dim=1
  )
  assert result.tolist() == [
      [0.0, 1.0, 2.0, 3.0],
      [10.0, 11.0, 12.0, 13.0],
  ]


def test_padded_gather_validates_local_size(monkeypatch):
  shard = _ragged_shard()
  monkeypatch.setattr(dist, "get_rank", lambda: 1)
  with pytest.raises(ValueError, match="expected 1"):
    shard.all_gather_padded_into_tensor(input_tensor=torch.empty(2, 3))


def test_padded_gather_fake_tensor_is_metadata_only(monkeypatch):
  shard = _ragged_shard()
  monkeypatch.setattr(dist, "get_rank", lambda: 1)

  def unexpected_collective(*args, **kwargs):
    raise AssertionError("FakeTensor gather must not enter the backend")

  monkeypatch.setattr(
      dist, "all_gather_into_tensor", unexpected_collective
  )
  from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

  with FakeTensorMode():
    result = shard.all_gather_padded_into_tensor(
        input_tensor=torch.empty(2, 1, 3), dim=1
    )
  assert is_fake(result)
  assert result.shape == (2, 4, 3)
