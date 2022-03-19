# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
from dataclasses import dataclass, fields
from typing import Any

import hypothesis.strategies as st
import pytest
import torch as tr
from hypothesis import assume, given
from omegaconf import OmegaConf
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset

from hydra_zen import builds, hydrated_dataclass, instantiate, just, to_yaml
from hydra_zen.structured_configs._utils import safe_name

torch_objects = [
    tr.softmax,
    tr.tensor,
    tr.randn,
    tr.is_tensor,
    tr.as_strided,
    tr.Tensor,
    tr.dtype,
    tr.empty,
    tr.optim.Adam,
    tr.nn.Linear,
    tr.nn.ReLU,
    tr.nn.Sequential,
    tr.nn.BatchNorm1d,
    tr.autograd.backward,
    tr.autograd.grad,
    tr.autograd.functional.jacobian,
    tr.cuda.current_device,
    tr.cuda.device_count,
    tr.backends.cuda.is_built,  # type: ignore
    tr.distributed.is_available,  # type: ignore
    tr.distributions.bernoulli.Bernoulli,
    tr.fft.fft,
    tr.jit.script,  # type: ignore
    tr.linalg.cholesky,
    tr.overrides.get_ignored_functions,  # type: ignore
    tr.profiler.profile,  # type: ignore
    DataLoader,
]


@pytest.mark.parametrize("obj", torch_objects)
def test_just_roundtrip(obj):
    assert instantiate(just(obj)) is obj


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (Adam, "Adam"),
    ],
)
def test_safename_known(obj, expected_name):
    assert safe_name(obj) == expected_name


@pytest.mark.parametrize("target", torch_objects)
@given(partial=st.booleans(), full_sig=st.booleans())
def test_fuzz_build_validation_against_a_bunch_of_common_objects(
    target, partial: bool, full_sig: bool
):
    doesnt_have_sig = False
    try:
        inspect.signature(target)
    except ValueError:
        doesnt_have_sig = True

    if doesnt_have_sig and full_sig:
        assume(False)
    conf = builds(target, zen_partial=partial, populate_full_signature=full_sig)

    OmegaConf.create(to_yaml(conf))  # ensure serializable

    if partial:
        instantiate(conf)  # ensure instantiable


def test_documented_builds_roundtrip_partial_with_interpolation_example():
    import torch as tr
    from torch.optim import Adam

    @dataclass
    class ModuleConfig:
        learning_rate: float = 100.2
        optimizer: Any = builds(Adam, lr="${learning_rate}", zen_partial=True)

    params = [tr.tensor(1.0)]
    config = instantiate(ModuleConfig)
    optim = config.optimizer(params)
    assert isinstance(optim, Adam)
    assert optim.defaults["lr"] == 100.2


def test_documented_inheritance_example():
    @dataclass
    class AdamBaseConfig:
        lr: float = 0.001
        eps: float = 1e-8

    @hydrated_dataclass(target=AdamW, zen_partial=True)
    class AdamWConfig(AdamBaseConfig):
        weight_decay: float = 0.01

    partialed = instantiate(AdamWConfig)
    assert partialed.keywords == {"lr": 0.001, "eps": 1e-08, "weight_decay": 0.01}
    assert partialed.func is AdamW


def test_parse_dataloader_sig():
    with pytest.raises(TypeError):
        # should be `batch_size` not `batch_sizes`
        builds(DataLoader, batch_sizes=2)


# DataSet and DataLoader both inherit from `typing.Generic`, which obfuscates
# their signatures
# >>> inspect.signature(DataLoader)
# Signature(*args, **kwargs)


class CustomDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return tr.tensor((float(index)))

    def __len__(self):
        return 5


def test_dataloader_sig_doesnt_have_self():
    Conf = builds(
        DataLoader, dataset=builds(CustomDataSet), populate_full_signature=True
    )
    _fields = fields(Conf)
    assert "self" not in {f.name for f in _fields}
    assert len(_fields) > 2
    assert list(instantiate(Conf)) == [tr.tensor(float(i)) for i in range(5)]
