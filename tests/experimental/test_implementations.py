# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import OmegaConf

from hydra_zen import builds, instantiate
from hydra_zen.experimental import hydra_multirun, hydra_run
from hydra_zen.experimental._implementations import _load_config, _store_config


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
    [None, [], ["hydra.run.dir=test_hydra_overrided"]],
)
@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize(
    "as_dictconfig, with_hydra", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("use_default_dir", [True, False])
def test_hydra_run_job(
    overrides, as_dataclass, as_dictconfig, with_hydra, use_default_dir
):
    if not as_dataclass:
        cfg = dict(a=1, b=1)
    else:
        cfg = builds(dict, a=1, b=1)

    override_exists = overrides and len(overrides) > 1

    if as_dictconfig:
        if not with_hydra:
            cfg = OmegaConf.create(cfg)
        else:
            cn = _store_config(cfg)
            cfg = _load_config(cn, overrides=overrides)
            overrides = []

    additl_kwargs = {} if use_default_dir else dict(config_dir=Path.cwd())
    job = hydra_run(
        cfg, task_function=instantiate, overrides=overrides, **additl_kwargs
    )
    assert job.return_value == {"a": 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides",
    [None, [], ["hydra.sweep.dir=test_hydra_overrided"]],
)
@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize(
    "as_dictconfig, with_hydra", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("use_default_dir", [True, False])
def test_hydra_multirun(
    overrides, as_dataclass, as_dictconfig, with_hydra, use_default_dir
):
    if not as_dataclass:
        cfg = dict(a=1, b=1)
    else:
        cfg = builds(dict, a=1, b=1)
    multirun_overrides = ["a=1,2"]
    override_exists = overrides and len(overrides) > 1

    if as_dictconfig:
        if not with_hydra:
            cfg = OmegaConf.create(cfg)
        else:
            cn = _store_config(cfg)
            cfg = _load_config(cn, overrides=overrides)
            overrides = []

    additl_kwargs = {} if use_default_dir else dict(config_dir=Path.cwd())
    _overrides = (
        multirun_overrides if overrides is None else (overrides + multirun_overrides)
    )
    job = hydra_multirun(
        cfg, task_function=instantiate, overrides=_overrides, **additl_kwargs
    )
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides",
    [["hydra/launcher=submitit_local"]],
)
def test_hydra_multirun_plugin(overrides):
    try:
        from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
            BaseSubmititLauncher,
        )
    except ImportError:
        pytest.skip("Submitit plugin not available")
        return

    cfg = builds(dict, a=1, b=1)
    multirun_overrides = ["a=1,2"]

    _overrides = (
        multirun_overrides if overrides is None else overrides + multirun_overrides
    )
    job = hydra_multirun(cfg, task_function=instantiate, overrides=_overrides)
    for i, j in enumerate(job[0]):
        submitit_folder = Path(j.working_dir).parent / ".submitit"
        assert submitit_folder.exists()
        assert j.return_value == {"a": i + 1, "b": 1}
