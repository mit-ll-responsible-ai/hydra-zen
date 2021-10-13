# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import OmegaConf

from hydra_zen import builds, instantiate, launch
from hydra_zen._launch import _store_config


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
@pytest.mark.parametrize("multirun", [False, True])
@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize(
    "as_dictconfig, with_hydra", [(True, True), (True, False), (False, False)]
)
def test_launch_config_type(
    multirun,
    as_dataclass,
    as_dictconfig,
    with_hydra,
):
    if not as_dataclass:
        cfg = dict(a=1, b=1)
    else:
        cfg = builds(dict, a=1, b=1)

    if as_dictconfig:
        if not with_hydra:
            cfg = OmegaConf.create(cfg)
        else:
            cn = _store_config(cfg)
            with initialize(config_path=None):
                cfg = compose(config_name=cn)

    job = launch(cfg, task_function=instantiate, multirun=multirun)
    if isinstance(job, list):
        job = job[0][0]

    assert job.return_value == {"a": 1, "b": 1}


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides", [None, [], ["hydra.run.dir=test_hydra_overrided"]]
)
@pytest.mark.parametrize("config_dir", [Path.cwd(), None])
@pytest.mark.parametrize("with_log_configuration", [False, True])
def test_launch_job(
    overrides,
    config_dir,
    with_log_configuration,
):
    cfg = dict(a=1, b=1)
    override_exists = overrides and len(overrides) > 1

    job = launch(
        cfg,
        task_function=instantiate,
        overrides=overrides,
        config_dir=config_dir,
        with_log_configuration=with_log_configuration,
    )
    assert job.return_value == {"a": 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides", [None, [], ["hydra.sweep.dir=test_hydra_overrided"]]
)
@pytest.mark.parametrize("multirun_overrides", [None, ["a=1,2"]])
@pytest.mark.parametrize("config_dir", [Path.cwd(), None])
@pytest.mark.parametrize("with_log_configuration", [False, True])
def test_hydra_multirun(
    overrides,
    multirun_overrides,
    config_dir,
    with_log_configuration,
):
    cfg = dict(a=1, b=1)
    override_exists = overrides and len(overrides) > 1

    _overrides = overrides
    if multirun_overrides is not None:
        _overrides = (
            multirun_overrides
            if overrides is None
            else (overrides + multirun_overrides)
        )

    job = launch(
        cfg,
        task_function=instantiate,
        overrides=_overrides,
        config_dir=config_dir,
        with_log_configuration=with_log_configuration,
        multirun=True
    )
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()



from tests import MyBasicSweeper
from hydra.errors import ConfigCompositionException
cs = ConfigStore.instance()
cs.store(group="hydra/sweeper", name="local_test", node=builds(MyBasicSweeper, max_batch_size=None))

@pytest.mark.usefixtures("cleandir")
def test_launch_with_multirun_overrides():
    cfg = builds(dict, a=1, b=1)
    multirun_overrides = ["hydra/sweeper=basic", "a=1,2"]
    with pytest.raises(ConfigCompositionException):
        launch(cfg, instantiate, overrides=multirun_overrides)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "plugin",
    [["hydra/sweeper=basic", "hydra/sweeper=local_test"]],
)
def test_launch_with_multirun_plugin(plugin):
    cfg = builds(dict, a=1, b=1)
    multirun_overrides = plugin + ["a=1,2"]
    job = launch(cfg, instantiate, overrides=multirun_overrides, multirun=True)
    assert isinstance(job, list) and len(job) == 1 and len(job[0]) == 2
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}
