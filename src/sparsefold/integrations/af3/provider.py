# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Fixed-version official AF3 boundary; core inference never imports this module."""
from dataclasses import dataclass, fields
import datetime
import base64
import json
import hashlib
import importlib
import importlib.util
import importlib.metadata
import importlib.machinery
from urllib.parse import urlparse, unquote
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .compatibility import DEFAULT_AF3_REVISION, SUPPORTED_REVISIONS, SCHEMA_ID

# Preserve the original public export while keeping one compatibility authority.
APACHE_REVISION = DEFAULT_AF3_REVISION

class ProviderError(RuntimeError):
    pass

@dataclass(frozen=True)
class PreparedExample:
    seed: int
    model_batch: Any
    output_context: Any
    provider_fingerprint: Mapping[str, Any]

class AF3Provider:
    def __init__(self, revision=APACHE_REVISION, source_root=None):
        self.revision = revision
        self.source_root = Path(source_root).resolve() if source_root is not None else None
        self._fingerprint = None

    def _git(self, root, *args):
        result=subprocess.run(['git','-C',str(root),*args],check=False,text=True,capture_output=True)
        if result.returncode: raise ProviderError(f'Cannot verify official AF3 checkout: {result.stderr.strip()}')
        return result.stdout.strip()

    @staticmethod
    def _file_digest(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest

    def _native_artifacts(self, root):
        """Trust only the editable distribution's recorded official cpp binary."""
        try:
            distribution = importlib.metadata.distribution('alphafold3')
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProviderError('Official AF3 editable distribution metadata is missing') from exc
        try:
            direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
            location = urlparse(direct_url.get('url', ''))
            installed_root = Path(unquote(location.path)).resolve()
        except (ValueError, TypeError) as exc:
            raise ProviderError('Invalid official AF3 editable direct_url metadata') from exc
        if location.scheme != 'file' or location.netloc or not direct_url.get('dir_info', {}).get('editable') or installed_root != root:
            raise ProviderError(f'AF3 distribution is not an editable install of the pinned checkout {root}')
        artifacts = {}
        for recorded in distribution.files or ():
            file = Path(str(recorded))
            if file.parent != Path('alphafold3') or not file.name.startswith('cpp.'):
                continue
            if not any(file.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                continue
            actual = Path(distribution.locate_file(recorded)).resolve()
            record_hash = recorded.hash
            if not actual.is_file() or record_hash is None or record_hash.mode != 'sha256':
                raise ProviderError(f'Official AF3 native extension has no valid SHA256 RECORD: {actual}')
            digest = self._file_digest(actual).digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
            if encoded != record_hash.value:
                raise ProviderError(f'Official AF3 native extension RECORD checksum mismatch: {actual}')
            artifacts[actual] = digest.hex()
        if not artifacts:
            raise ProviderError('Official AF3 editable distribution contains no recorded cpp extension')
        return artifacts

    def _check_origin(self, name, location, package, native_artifacts):
        if not location:
            return
        location = Path(location).resolve()
        if location.is_relative_to(package):
            return
        if name == 'alphafold3.cpp' and location in native_artifacts:
            return
        raise ProviderError(f'AF3 module {name} is shadowed by {location}')

    def verify(self):
        if self.revision not in SUPPORTED_REVISIONS:
            raise ProviderError(f'Unsupported AF3 revision {self.revision}; supported: {sorted(SUPPORTED_REVISIONS)}. Add a reviewed schema profile before using another revision.')
        spec=importlib.util.find_spec('alphafold3')
        if spec is None or not spec.origin:
            raise ProviderError('Official AF3 must be installed separately from the fixed source checkout; no automatic install is performed.')
        package=Path(spec.origin).resolve().parent
        inferred=package.parent.parent
        root=self.source_root or inferred
        if package != root/'src'/'alphafold3':
            raise ProviderError(f'AF3 namespace shadowed: resolves to {package}, expected {root / "src/alphafold3"}')
        if self._git(root,'rev-parse','--show-toplevel') != str(root):
            raise ProviderError('AF3 source_root is not the checkout root')
        head=self._git(root,'rev-parse','HEAD')
        if head != self.revision:
            raise ProviderError(f'AF3 version mismatch: expected {self.revision}, found {head}')
        dirty=self._git(root,'status','--porcelain','--untracked-files=no')
        if dirty: raise ProviderError(f'AF3 checkout has tracked modifications; use a clean verified checkout. Changed files: {dirty}')
        if (package/'torch').exists():
            raise ProviderError('Legacy PyTorch alphafold3 fork rejected; install official AF3 in its own namespace/environment')
        native_artifacts = self._native_artifacts(root)
        for name,mod in list(sys.modules.items()):
            if name=='alphafold3' or name.startswith('alphafold3.'):
                location=getattr(mod,'__file__',None)
                self._check_origin(name,location,package,native_artifacts)
        digest=hashlib.sha256()
        paths=self._git(root,'ls-files','-z','--','src/alphafold3','pyproject.toml','CMakeLists.txt').split('\0')
        for name in sorted(p for p in paths if p):
            path=root/name
            if path.is_file():
                digest.update(name.encode());digest.update(b'\0');digest.update(path.read_bytes())
        self._fingerprint={'revision':head,'source_tree_sha256':digest.hexdigest(),'schema_id':SCHEMA_ID,'source_root':str(root),'native_artifacts':{path.name:digest for path,digest in native_artifacts.items()}}
        return dict(self._fingerprint)

    def module(self,name):
        if name != 'alphafold3' and not name.startswith('alphafold3.'):
            raise ProviderError(f'Only official AF3 modules can be requested: {name}')
        self.verify()
        mod=importlib.import_module(name)
        location=getattr(mod,'__file__',None)
        package=Path(self._fingerprint['source_root'])/'src/alphafold3'
        self._check_origin(name,location,package,self._native_artifacts(Path(self._fingerprint['source_root'])))
        return mod

    def read_inputs(self,path):
        path=Path(path)
        module=self.module('alphafold3.common.folding_input')
        loader=module.load_fold_inputs_from_dir if path.is_dir() else module.load_fold_inputs_from_path
        return list(loader(path))

    def prepare(self,fold_input,config):
        """Return enriched official Input, distinct from numerical featurisation."""
        pipeline=self.module('alphafold3.data.pipeline')
        if isinstance(config,pipeline.DataPipelineConfig): return pipeline.DataPipeline(config).process(fold_input)
        config=dict(config)
        valid={f.name for f in fields(pipeline.DataPipelineConfig)}
        legacy={key for key in config if key.endswith('_z_value') or key.endswith('_max_parallel_shards')}
        for key in legacy-valid:
            if config[key] is not None:
                raise ProviderError(f'Custom legacy data option {key}={config[key]!r} has no official AF3 equivalent; provide enriched input externally or choose a supported upstream setting.')
            del config[key]
        unknown=config.keys()-valid
        if unknown: raise ProviderError(f'Unsupported official DataPipelineConfig options: {sorted(unknown)}')
        if isinstance(config.get('max_template_date'),str): config['max_template_date']=datetime.date.fromisoformat(config['max_template_date'])
        return pipeline.DataPipeline(pipeline.DataPipelineConfig(**config)).process(fold_input)

    def featurise(self,fold_input,*,ccd=None,ccd_path=None,buckets=None,**options):
        from sparsefold.schema.batch import ModelBatch
        from sparsefold.schema.context import OutputContext
        from sparsefold.schema.validation import validate_mapping
        self.verify()
        if ccd is None:
            ccd_module=self.module('alphafold3.constants.chemical_components')
            ccd=ccd_module.Ccd(ccd_pickle_path=ccd_path,user_ccd=fold_input.user_ccd)
        featurisation=self.module('alphafold3.data.featurisation')
        import inspect
        unknown=options.keys()-inspect.signature(featurisation.featurise_input).parameters.keys()
        if unknown: raise ProviderError(f'Unsupported official featurisation options: {sorted(unknown)}')
        fingerprint = dict(self._fingerprint)
        ccd_file = getattr(ccd, '_ccd_pickle_path', None)
        if ccd_file is None or not Path(ccd_file).is_file():
            raise ProviderError('CCD provenance unavailable; pass the pinned official Ccd with an existing local CCD resource')
        fingerprint['ccd_sha256'] = self._file_digest(ccd_file).hexdigest()
        resources = self.module('alphafold3.common.resources')
        component_sets_file = Path(resources.filename('constants/converters/chemical_component_sets.pickle'))
        if not component_sets_file.is_file():
            raise ProviderError('Official chemical component sets are missing; complete official build_data in the isolated environment')
        fingerprint['chemical_component_sets_sha256'] = self._file_digest(component_sets_file).hexdigest()
        fingerprint['user_ccd_sha256'] = hashlib.sha256((fold_input.user_ccd or '').encode()).hexdigest()
        fingerprint['featurisation'] = {'buckets': None if buckets is None else list(buckets), 'options': {key: value.isoformat() if isinstance(value, datetime.date) else value for key, value in options.items()}}
        raw=featurisation.featurise_input(fold_input,ccd,buckets,**options)
        seeds=list(fold_input.rng_seeds)
        if len(raw)!=len(seeds): raise ProviderError('Official AF3 returned different batch/seed counts')
        prepared=[]
        for seed,data in zip(seeds,raw):
            validate_mapping(data,strict=True)
            context=OutputContext.from_data_dict(data)
            if context.token_atoms_layout is None or context.flat_output_layout is None or context.empty_output_struc is None:
                raise ProviderError('Official output context lacks required atom layouts/empty structure')
            batch=ModelBatch.from_data_dict(data)
            from sparsefold.integrations.af3.artifact import validate_context
            validate_context(context,batch)
            if tuple(context.token_atoms_layout.shape)!=tuple(batch.predicted_structure_info.atom_mask.shape):
                raise ProviderError('Official atom identity layout does not match numerical dense atom shape')
            prepared.append(PreparedExample(int(seed),batch,context,fingerprint))
        return prepared
