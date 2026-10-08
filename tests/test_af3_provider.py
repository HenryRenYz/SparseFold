# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
import pytest
from sparsefold.integrations.af3.provider import AF3Provider,APACHE_REVISION,ProviderError


def test_provider_rejects_unreviewed_revision():
    with pytest.raises(ProviderError,match='Unsupported AF3 revision'):AF3Provider(revision='unknown').verify()


def test_provider_rejects_namespace_shadowing(monkeypatch,tmp_path):
    monkeypatch.setattr('importlib.util.find_spec',lambda name:SimpleNamespace(origin=str(tmp_path/'old/alphafold3/__init__.py')))
    with pytest.raises(ProviderError,match='shadowed'):AF3Provider(source_root=tmp_path/'official').verify()


def test_provider_rejects_missing_package(monkeypatch):
    monkeypatch.setattr('importlib.util.find_spec',lambda name:None)
    with pytest.raises(ProviderError,match='installed separately'):AF3Provider().verify()


def test_provider_rejects_non_af3_imports():
    with pytest.raises(ProviderError,match='Only official'):AF3Provider().module('torch')


def distribution_fixture(monkeypatch,tmp_path,*,editable=True,checksum=True):
    import base64,hashlib,json
    import importlib.metadata
    import importlib.machinery
    root=tmp_path/'checkout';root.mkdir()
    site=tmp_path/'site-packages';site.mkdir()
    name='cpp'+importlib.machinery.EXTENSION_SUFFIXES[0]
    binary=site/'alphafold3'/name;binary.parent.mkdir();binary.write_bytes(b'official-native-fixture')
    entry=importlib.metadata.PackagePath('alphafold3/'+name)
    value=base64.urlsafe_b64encode(hashlib.sha256(binary.read_bytes()).digest()).rstrip(b'=').decode()
    entry.hash=SimpleNamespace(mode='sha256',value=value if checksum else 'invalid')
    distribution=SimpleNamespace(files=[entry],locate_file=lambda path:site/str(path),read_text=lambda name:json.dumps({'url':root.as_uri(),'dir_info':{'editable':editable}}))
    monkeypatch.setattr(importlib.metadata,'distribution',lambda name:distribution)
    return root,binary


def test_native_origin_requires_exact_editable_distribution_and_record(monkeypatch,tmp_path):
    root,binary=distribution_fixture(monkeypatch,tmp_path)
    provider=AF3Provider(source_root=root)
    artifacts=provider._native_artifacts(root)
    assert set(artifacts)=={binary}
    provider._check_origin('alphafold3.cpp',str(binary),root/'src/alphafold3',artifacts)
    with pytest.raises(ProviderError,match='shadowed'):
        provider._check_origin('alphafold3.model.model',str(binary),root/'src/alphafold3',artifacts)
    with pytest.raises(ProviderError,match='shadowed'):
        provider._check_origin('alphafold3.cpp',str(binary.parent/'unrecorded.so'),root/'src/alphafold3',artifacts)


@pytest.mark.parametrize('editable,checksum,match',[(False,True,'not an editable'),(True,False,'checksum mismatch')])
def test_native_origin_rejects_wrong_distribution_or_tampered_binary(monkeypatch,tmp_path,editable,checksum,match):
    root,binary=distribution_fixture(monkeypatch,tmp_path,editable=editable,checksum=checksum)
    with pytest.raises(ProviderError,match=match):AF3Provider()._native_artifacts(root)
