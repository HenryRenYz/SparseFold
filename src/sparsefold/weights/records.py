# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Model param loading."""

import bisect
import collections
import contextlib
import io
import os
import pathlib
import re
import struct
import sys
from collections.abc import Iterator
from typing import IO
from dataclasses import dataclass
import ml_dtypes  # for bfloat16
import numpy as np
import torch
import zstandard

Params = dict[str, dict[str, torch.Tensor]]


class RecordError(Exception):
  """Error reading a record."""


def encode_record(scope: str, name: str, arr: np.ndarray) -> bytes:
  """Encodes a single haiku param as bytes, preserving non-numpy dtypes."""
  scope = scope.encode('utf-8')
  name = name.encode('utf-8')
  shape = arr.shape
  dtype = str(arr.dtype).encode('utf-8')
  arr = np.ascontiguousarray(arr)
  if sys.byteorder == 'big':
    arr = arr.byteswap()
  arr_buffer = arr.tobytes('C')
  header = struct.pack(
    '<5i', len(scope), len(name), len(dtype), len(shape), len(arr_buffer)
  )
  return header + b''.join(
    (scope, name, dtype, struct.pack(f'<{len(shape)}i', *shape), arr_buffer)
  )


def _read_record(stream: IO[bytes]) -> tuple[str, str, torch.Tensor] | None:
  """Reads a record encoded by `_encode_record` from a byte stream."""
  header_size = struct.calcsize('<5i')
  header = stream.read(header_size)
  if not header:
    return None
  if len(header) < header_size:
    raise RecordError(f'Incomplete header: {len(header)=} < {header_size=}')
  (scope_len, name_len, dtype_len, shape_len, arr_buffer_len) = struct.unpack(
    '<5i', header
  )
  if min(scope_len, name_len, dtype_len, shape_len, arr_buffer_len) < 0:
    raise RecordError('Negative record length')
  if shape_len > 64 or scope_len + name_len + dtype_len > 1_048_576:
    raise RecordError('Unreasonable record header length')
  fmt = f'<{scope_len}s{name_len}s{dtype_len}s{shape_len}i'
  payload_size = struct.calcsize(fmt) + arr_buffer_len
  payload = stream.read(payload_size)
  if len(payload) < payload_size:
    raise RecordError(f'Incomplete payload: {len(payload)=} < {payload_size=}')
  try:
    scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
    scope = scope.decode('utf-8')
    name = name.decode('utf-8')
    dtype = dtype.decode('utf-8')
    numpy_dtype = np.dtype(dtype)
    if numpy_dtype.hasobject or (dtype != 'bfloat16' and numpy_dtype.kind in ('S', 'U', 'V')):
      raise ValueError('Unsupported object/string/structured tensor dtype')
    if any(dim < 0 for dim in shape):
      raise ValueError('Negative tensor dimension')
    import math
    if math.prod(shape) * numpy_dtype.itemsize != arr_buffer_len:
      raise ValueError('Tensor shape, dtype and payload length disagree')
    raw = payload[struct.calcsize(fmt):]
    # Reading uint16 preserves the bfloat16 bits without interpreting them as fp32.
    if dtype == 'bfloat16':
      arr = np.frombuffer(raw, dtype='<u2').copy()
      tensor = torch.from_numpy(arr).view(torch.bfloat16)
    else:
      arr = np.frombuffer(raw, dtype=numpy_dtype.newbyteorder('<')).copy()
      if sys.byteorder == 'big':
        arr = arr.byteswap().view(numpy_dtype.newbyteorder('='))
      tensor = torch.from_numpy(arr)
    return scope, name, tensor.reshape(shape)
  except (ValueError, TypeError, UnicodeError, RuntimeError) as exc:
    raise RecordError(f'Invalid record: {exc}') from exc


def read_records(stream: IO[bytes]) -> Iterator[tuple[str, str, torch.Tensor]]:
  """Fully reads the contents of a byte stream."""
  while record := _read_record(stream):
    yield record


class _MultiFileIO(io.RawIOBase):
  """A file-like object that presents a concatenated view of multiple files."""

  def __init__(self, files: list[pathlib.Path]):
    self._files = files
    self._stack = contextlib.ExitStack()
    self._handles = [
      self._stack.enter_context(file.open('rb')) for file in files
    ]
    self._sizes = []
    for handle in self._handles:
      handle.seek(0, os.SEEK_END)
      self._sizes.append(handle.tell())
    self._length = sum(self._sizes)
    self._offsets = [0]
    for s in self._sizes[:-1]:
      self._offsets.append(self._offsets[-1] + s)
    self._abspos = 0
    self._relpos = self._abs_to_rel(0)

  def _abs_to_rel(self, pos: int) -> tuple[int, int]:
    idx = bisect.bisect_right(self._offsets, pos) - 1
    return idx, pos - self._offsets[idx]

  def close(self):
    self._stack.close()
    super().close()

  def fileno(self) -> int:
    return -1

  def readable(self) -> bool:
    return True

  def tell(self) -> int:
    return self._abspos

  def seek(self, pos: int, whence: int = os.SEEK_SET, /):
    match whence:
      case os.SEEK_SET:
        pass
      case os.SEEK_CUR:
        pos += self._abspos
      case os.SEEK_END:
        pos += self._length
      case _:
        raise ValueError(f'Invalid whence: {whence}')
    if pos < 0:
      raise ValueError('Negative seek position')
    self._abspos = pos
    self._relpos = self._abs_to_rel(pos)
    return pos

  def readinto(self, b: bytearray | memoryview) -> int:
    result = 0
    mem = memoryview(b)
    while mem and self._abspos < self._length:
      self._handles[self._relpos[0]].seek(self._relpos[1])
      count = self._handles[self._relpos[0]].readinto(mem)
      result += count
      self._abspos += count
      self._relpos = self._abs_to_rel(self._abspos)
      mem = mem[count:]
      if count == 0 and self._abspos < self._length:
        raise RecordError('Unexpected EOF in checkpoint shard')
      if self._abspos == self._length:
        break
    return result


@contextlib.contextmanager
def open_for_reading(model_files: list[pathlib.Path], is_compressed: bool):
  with contextlib.closing(_MultiFileIO(model_files)) as f:
    if is_compressed:
      with zstandard.ZstdDecompressor().stream_reader(f) as stream:
        yield stream
    else:
      yield f


def _match_model(
    paths: list[pathlib.Path], pattern: re.Pattern[str]
) -> dict[str, list[pathlib.Path]]:
  """Match files in a directory with a pattern, and group by model name."""
  models = collections.defaultdict(list)
  for path in paths:
    match = pattern.fullmatch(path.name)
    if match:
      models[match.group('model_name')].append(path)
  return {k: sorted(v, key=lambda p: tuple(int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name))) for k, v in models.items()}


def select_model_files(
    model_dir: pathlib.Path, model_name: str | None = None
) -> tuple[list[pathlib.Path], bool]:
  """Select the model files from a model directory."""
  files = [file for file in model_dir.iterdir() if file.is_file()]

  for pattern, is_compressed in (
      (r'(?P<model_name>.*)\.[0-9]+\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin\.zst\.[0-9]+$', True),
      (r'(?P<model_name>.*)\.[0-9]+\.bin$', False),
      (r'(?P<model_name>.*)\.bin\.[0-9]+$', False),
      (r'(?P<model_name>.*)\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin$', False),
  ):
    models = _match_model(files, re.compile(pattern))
    if model_name is not None:
      if model_name in models:
        return models[model_name], is_compressed
    else:
      if models:
        if len(models) > 1:
          raise RuntimeError(f'Multiple models matched in {model_dir}')
        _, model_files = models.popitem()
        return model_files, is_compressed
  raise FileNotFoundError(f'No models matched in {model_dir}')


def _normalize_scope(scope: str) -> str:
  """The legacy Haiku transparent-module convention, not an internal name rename."""
  return scope.replace('/~/', '/').replace('/~', '/')


@dataclass(frozen=True)
class Checkpoint:
  params: Params
  identifier: bytes | None
  source_scopes: dict[str, str]


def _load_records(model_path: pathlib.Path, model_name: str | None = None) -> Params:
  if model_path.is_dir():
    model_files, compressed = select_model_files(model_path, model_name)
  elif model_path.is_file():
    if model_name is not None:
      raise ValueError('model_name applies only to a directory')
    if not (model_path.name.endswith('.bin') or model_path.name.endswith('.bin.zst')):
      raise ValueError('Explicit checkpoint file must end in .bin or .bin.zst')
    model_files, compressed = [model_path], model_path.name.endswith('.zst')
  else:
    raise FileNotFoundError(model_path)
  params: Params = {}
  with open_for_reading(model_files, compressed) as stream:
    for scope, name, tensor in read_records(stream):
      target = params.setdefault(scope, {})
      if name in target:
        raise RecordError(f'Duplicate checkpoint tensor: {scope}/{name}')
      target[name] = tensor
  if not params:
    raise RecordError(f'Empty checkpoint: {model_path}')
  return params


def read_checkpoint(model_path: pathlib.Path | str, model_name: str | None = None) -> Checkpoint:
  """Read local AF3 records without loading executable pickle or downloading weights."""
  original = _load_records(pathlib.Path(model_path), model_name)
  params: Params = {}
  scopes: dict[str, str] = {}
  identifier = None
  for raw_scope, tensors in original.items():
    scope = _normalize_scope(raw_scope)
    if scope in scopes:
      raise RecordError(f'Normalized scope collision: {scopes[scope]!r}, {raw_scope!r}')
    scopes[scope] = raw_scope
    if scope == '__meta__':
      if set(tensors) != {'__identifier__'}:
        raise RecordError(f'Unknown checkpoint metadata: {sorted(tensors)}')
      tensor = tensors['__identifier__']
      if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise RecordError('Checkpoint identifier must be a one-dimensional uint8 tensor')
      identifier = tensor.numpy().tobytes()
    else:
      params[scope] = tensors
  return Checkpoint(params, identifier, scopes)


def get_model_params(model_dir: pathlib.Path) -> Params:
  """Compatibility API including the legacy metadata scope."""
  original = _load_records(pathlib.Path(model_dir))
  return _wash_params_path(original)


def _wash_params_path(params: Params) -> Params:
  result: Params = {}
  for raw_scope, tensors in params.items():
    scope = _normalize_scope(raw_scope)
    if scope in result:
      raise RecordError(f'Normalized scope collision: {scope}')
    result[scope] = tensors
  return result
