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
        multirun=True,
    )
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


from hydra.errors import ConfigCompositionException


@pytest.mark.usefixtures("cleandir")
def test_launch_with_multirun_overrides():
    cfg = builds(dict, a=1, b=1)
    multirun_overrides = ["hydra/sweeper=basic", "a=1,2"]
    with pytest.raises(ConfigCompositionException):
        launch(cfg, instantiate, overrides=multirun_overrides)


###############################################
# Test local plugins work with hydra_zen.launch
###############################################

import itertools

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import HydraException
from hydra.plugins.sweeper import Sweeper


class LocalBasicSweeper(Sweeper):
    def __init__(self):
        super().__init__()
        self.overrides = None
        self.batch_index = 0

        self.hydra_context = None
        self.config = None
        self.launcher = None

    def setup(self, *, hydra_context, task_function, config):
        from hydra.core.plugins import Plugins

        self.hydra_context = hydra_context
        self.config = config

        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    @staticmethod
    def split_overrides_to_chunks(lst, n):
        if n is None or n == -1:
            n = len(lst)
        assert n > 0
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    @staticmethod
    def split_arguments(overrides):

        lists = []
        for override in overrides:
            if override.is_sweep_override():
                if override.is_discrete_sweep():
                    key = override.get_key_element()
                    sweep = [f"{key}={val}" for val in override.sweep_string_iterator()]
                    lists.append(sweep)
                else:
                    assert override.value_type is not None
                    raise HydraException(
                        f"{LocalBasicSweeper.__name__} does not support sweep type : {override.value_type.name}"
                    )
            else:
                key = override.get_key_element()
                value = override.get_value_element_as_str()
                lists.append([f"{key}={value}"])

        return [[list(x) for x in itertools.product(*lists)]]

    def sweep(self, arguments):
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(arguments)

        self.overrides = self.split_arguments(overrides)
        returns = []

        initial_job_idx = 0
        while not self.is_done():
            batch = self.get_job_batch()
            self.validate_batch_is_legal(batch)
            results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)

            for r in results:
                _ = r.return_value

            initial_job_idx += len(batch)
            returns.append(results)

        return returns

    def get_job_batch(self):
        assert self.overrides is not None
        self.batch_index += 1
        return self.overrides[self.batch_index - 1]

    def is_done(self) -> bool:
        assert self.overrides is not None
        return self.batch_index >= len(self.overrides)


cs = ConfigStore.instance()
cs.store(group="hydra/sweeper", name="local_test", node=builds(LocalBasicSweeper))


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "plugin",
    [["hydra/sweeper=basic"], ["hydra/sweeper=local_test"]],
)
def test_launch_with_multirun_plugin(plugin):
    cfg = builds(dict, a=1, b=1)
    multirun_overrides = plugin + ["a=1,2"]
    job = launch(cfg, instantiate, overrides=multirun_overrides, multirun=True)
    assert isinstance(job, list) and len(job) == 1 and len(job[0]) == 2
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}
