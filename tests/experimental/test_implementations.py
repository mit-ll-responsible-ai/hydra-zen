# Copyright (c) 2021 Massachusetts Institute of Technology
from pathlib import Path

import pytest
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import OmegaConf

from hydra_utils import builds, instantiate
from hydra_utils.experimental import hydra_launch
from hydra_utils.experimental._implementations import _load_config, _store_config


@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize("as_dictconfig", [True, False])
def test_store_config(as_dataclass, as_dictconfig):
    cfg = builds(dict, a=1, b=1)

    if not as_dataclass:
        cfg = dict(f=cfg)

    if as_dictconfig:
        cfg = OmegaConf.create(cfg)

    cn = _store_config(cfg)
    cs = ConfigStore.instance()
    key = cn + ".yaml"
    assert key in cs.repo
    assert cs.repo[key].node == OmegaConf.create(cfg)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides",
    [[], ["hydra.run.dir=test_hydra_overrided"]],
)
@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize(
    "as_dictconfig, with_hydra", [(True, True), (True, False), (False, False)]
)
def test_hydra_launch_job(overrides, as_dataclass, as_dictconfig, with_hydra):
    cfg = builds(dict, a=1, b=1)
    task_function = lambda config: instantiate(config)
    override_exists = len(overrides) > 1

    if not as_dataclass:
        cfg = dict(f=cfg)
        task_function = lambda config: instantiate(config.f)

    if as_dictconfig:
        if not with_hydra:
            cfg = OmegaConf.create(cfg)
        else:
            cn = _store_config(cfg)
            cfg = _load_config(cn, overrides=overrides)
            overrides = []

    job = hydra_launch(
        cfg,
        task_function=task_function,
        overrides=overrides,
    )
    assert job.return_value == {"a": 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides",
    [[], ["hydra.sweep.dir=test_hydra_overrided"]],
)
@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize(
    "as_dictconfig, with_hydra", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("use_default_dir", [True, False])
def test_hydra_launch_multirun(
    overrides, as_dataclass, as_dictconfig, with_hydra, use_default_dir: bool
):
    cfg = builds(dict, a=1, b=1)
    task_function = lambda config: instantiate(config)
    multirun_overrides = ["a=1,2"]
    override_exists = len(overrides) > 1

    if not as_dataclass:
        cfg = dict(f=cfg)
        task_function = lambda config: instantiate(config.f)
        multirun_overrides = ["f.a=1,2"]

    if as_dictconfig:
        if not with_hydra:
            cfg = OmegaConf.create(cfg)
        else:
            cn = _store_config(cfg)
            cfg = _load_config(cn, overrides=overrides)
            overrides = []

    additl_kwargs = {} if use_default_dir else dict(config_dir=Path.cwd())
    job = hydra_launch(
        cfg,
        task_function=task_function,
        multirun_overrides=multirun_overrides,
        overrides=overrides,
        **additl_kwargs,
    )
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()
