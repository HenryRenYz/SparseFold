# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
import io
import struct

import ml_dtypes
import numpy as np
import pytest
import torch
import zstandard

from sparsefold.weights.records import (
    RecordError, _MultiFileIO, encode_record, read_checkpoint, read_records,
    select_model_files,
)


@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (2, 3)])
@pytest.mark.parametrize('dtype', [np.float32, np.int32, np.uint8, ml_dtypes.bfloat16])
def test_record_roundtrip_shape_and_bits(shape, dtype):
    array = np.arange(np.prod(shape) if shape else 1).reshape(shape).astype(dtype)
    data = encode_record('scope', 'tensor', array)
    [(scope, name, tensor)] = list(read_records(io.BytesIO(data)))
    assert (scope, name, tuple(tensor.shape)) == ('scope', 'tensor', shape)
    raw = b'' if tensor.numel() == 0 else tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    assert raw == array.tobytes()


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('layout', ['before', 'after'])
def test_numeric_shards_and_metadata(tmp_path, compressed, layout):
    data = encode_record('diffuser/~/linear', 'weights', np.arange(12, dtype=np.float32).reshape(3,4))
    data += encode_record('__meta__', '__identifier__', np.frombuffer(b'local synthetic', dtype=np.uint8))
    if compressed:
        data = zstandard.ZstdCompressor().compress(data)
    # Individual records and compressed frames cross arbitrary shard boundaries.
    parts = [data[i:i+max(1,len(data)//12)] for i in range(0, len(data), max(1,len(data)//12))]
    ext = '.bin.zst' if compressed else '.bin'
    for i, part in enumerate(parts):
        name = f'model.{i}{ext}' if layout == 'before' else f'model{ext}.{i}'
        (tmp_path / name).write_bytes(part)
    files, actual_compressed = select_model_files(tmp_path)
    assert actual_compressed == compressed
    assert files[2].name.endswith(f'.2{ext}') if layout == 'before' else files[2].name.endswith('.2')
    checkpoint = read_checkpoint(tmp_path)
    assert checkpoint.identifier == b'local synthetic'
    assert checkpoint.source_scopes['diffuser/linear'] == 'diffuser/~/linear'
    assert '__meta__' not in checkpoint.params
    assert torch.equal(checkpoint.params['diffuser/linear']['weights'], torch.arange(12).float().reshape(3,4))


def test_multifile_seek_and_empty_shard(tmp_path):
    paths = [tmp_path / str(i) for i in range(4)]
    for path, data in zip(paths, [b'', b'abc', b'', b'def']):
        path.write_bytes(data)
    with _MultiFileIO(paths) as stream:
        assert stream.read(20) == b'abcdef'
        assert stream.read(1) == b''
        assert stream.seek(-2, 2) == 4
        assert stream.read(5) == b'ef'
        with pytest.raises(ValueError, match='Negative seek'):
            stream.seek(-1)


@pytest.mark.parametrize('data', [b'bad', struct.pack('<5i', -1, 0, 0, 0, 0),
                                    struct.pack('<5i', 0,0,0,100000000,0)])
def test_malformed_header(data):
    with pytest.raises(RecordError):
        list(read_records(io.BytesIO(data)))


def test_duplicate_and_scope_collision(tmp_path):
    path = tmp_path/'model.bin'
    record = encode_record('diffuser/x', 'weights', np.zeros((2,3), np.float32))
    path.write_bytes(record + record)
    with pytest.raises(RecordError, match='Duplicate'):
        read_checkpoint(path)
    path.write_bytes(record + encode_record('diffuser/~/x','bias',np.zeros((3,), np.float32)))
    with pytest.raises(RecordError, match='collision'):
        read_checkpoint(path)


def test_invalid_payload_size():
    data = encode_record('x','y',np.ones(2,np.float32))
    # Alter shape to three while preserving declared bytes.
    prefix = struct.calcsize('<5i') + len('x') + len('y') + len('float32')
    data = data[:prefix] + struct.pack('<i',3) + data[prefix+4:]
    with pytest.raises(RecordError, match='payload length'):
        list(read_records(io.BytesIO(data)))
