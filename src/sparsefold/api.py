# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Public inference API; runtime selection applies to the calling process."""
from collections.abc import Mapping


def prepare_input(fold_input, config, *, provider):
    """Run the explicitly supplied official AF3 provider's retrieval stage."""
    return provider.prepare(fold_input, config)


def featurise_input(fold_input, *, provider, **options):
    """Return numeric ModelBatch and separate OutputContext for each seed."""
    return provider.featurise(fold_input, **options)


def _canonical_device(device):
    import torch
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f'Invalid SparseFold device: {device!r}') from error
    if selected.type == 'cpu':
        if selected.index not in (None, 0):
            raise ValueError('CPU device index must be absent or zero')
        return torch.device('cpu')
    if selected.type != 'cuda':
        raise ValueError(f'SparseFold inference supports CPU and CUDA, got {selected.type!r}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f'CUDA device index {index} is outside the visible device range')
    return torch.device('cuda', index)


def build_model_runner(*, model_dir, device='cpu', runtime_options: Mapping | None = None,
                       **model_options):
    """Configure runtime before model construction; weights remain lazy.

    Existing runtime/backend selections are preserved. ``runtime_options`` may
    override configure_runtime settings (including ``backend``), while the
    device is selected exclusively by ``device``. The unconfigured reference
    path defaults to Torch. Call prepare_model() to load and audit local weights.
    """
    from sparsefold.runtime import config
    from sparsefold import backends
    selected = _canonical_device(device)
    if runtime_options is not None and not isinstance(runtime_options, Mapping):
        raise TypeError('runtime_options must be a mapping of runtime settings')
    overrides = dict(runtime_options or {})
    if 'device' in overrides:
        raise ValueError('Select device through the device argument, not runtime_options')
    settings = config.get_runtime_options()
    if 'backend' in overrides:
        if 'acclib_implementation' in overrides:
            raise ValueError('Pass either backend or acclib_implementation')
        settings.pop('acclib_implementation')
    settings.update(overrides)
    settings['device'] = str(selected)
    config.configure_runtime(**settings)
    if selected.type == 'cuda':
        import torch
        torch.cuda.set_device(selected)
    backends.init()
    # Constructors and decorators resolve device/precision/backend from runtime.
    from sparsefold.cli.predict import ModelRunner, make_model_config
    return ModelRunner(make_model_config(**model_options), str(selected), model_dir)


def predict_batch(runner, batch, *, seed):
    """Predict a typed numeric ModelBatch; output decoding uses separate context."""
    from sparsefold.schema.batch import ModelBatch
    from sparsefold.runtime.util import manual_seed
    if not isinstance(batch, ModelBatch):
        raise TypeError('predict_batch requires ModelBatch; convert numerical mappings with ModelBatch.from_data_dict')
    manual_seed(seed)
    return runner.run_inference(batch.to_torch(device=runner._device))
