# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import dataclasses
from pathlib import Path
from typing import Optional

import pytest
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import ConfigCompositionException
from hydra.plugins.sweeper import Sweeper
from omegaconf.omegaconf import OmegaConf

from hydra_zen import builds, instantiate, launch, make_config
from hydra_zen._launch import _store_config

try:
    import cloudpickle

    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False

CONFIG_TYPE_EXAMPLES = [
    make_config(),
    builds(dict, a=1, b=1),
    dict(a=1),
    dict(f=builds(dict, a=1, b=1)),
    OmegaConf.create(dict(a=1)),
    OmegaConf.create(dict(f=builds(dict, a=1, b=1))),
]

DATACLASS_CONFIG_TYPE_EXAMPLES = [
    make_config(),
    builds(dict, a=1, b=1),
]


@pytest.mark.parametrize("cfg", CONFIG_TYPE_EXAMPLES)
def test_store_config(cfg):
    cn = _store_config(cfg)
    cs = ConfigStore.instance()
    key = cn + ".yaml"
    assert key in cs.repo
    assert cs.repo[key].node == OmegaConf.create(cfg)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("cfg", CONFIG_TYPE_EXAMPLES)
@pytest.mark.parametrize("multirun", [False, True])
def test_launch_config_type(cfg, multirun, version_base):
    job = launch(cfg, task_function=instantiate, multirun=multirun, **version_base)
    if isinstance(job, list):
        job = job[0][0]

    assert job.return_value == instantiate(cfg)


@pytest.mark.filterwarnings(
    "ignore:Your dataclass-based config was mutated by this run"
)
@pytest.mark.skipif(not CLOUDPICKLE_AVAIL, reason="cloudpickle not available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("cfg", DATACLASS_CONFIG_TYPE_EXAMPLES)
@pytest.mark.parametrize("to_dictconfig", [True, False])
def test_launch_to_dictconfig(cfg, to_dictconfig, version_base):
    pre_num_fields = len(dataclasses.fields(cfg))

    def task_fn(cfg):
        _ = cloudpickle.loads(cloudpickle.dumps(cfg))

    launch(cfg, task_function=task_fn, to_dictconfig=to_dictconfig, **version_base)

    if pre_num_fields > 0:
        if not to_dictconfig and int(cloudpickle.__version__.split(".")[0]) < 3:
            assert len(dataclasses.fields(cfg)) == 0
        else:
            assert len(dataclasses.fields(cfg)) > 0
    else:
        # run again with no error
        launch(cfg, task_function=task_fn, to_dictconfig=to_dictconfig, **version_base)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides", [None, [], ["hydra.run.dir=test_hydra_overrided"]]
)
@pytest.mark.parametrize("with_log_configuration", [False, True])
def test_launch_job(overrides, with_log_configuration, version_base):
    cfg = dict(a=1, b=1)
    override_exists = overrides and len(overrides) > 1

    job = launch(
        cfg,
        task_function=instantiate,
        overrides=overrides,
        with_log_configuration=with_log_configuration,
        **version_base,
    )
    assert job.return_value == {"a": 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides", [None, [], ["hydra.sweep.dir=test_hydra_overrided"]]
)
@pytest.mark.parametrize("multirun_overrides", [None, ["a=1,2"]])
@pytest.mark.parametrize("with_log_configuration", [False, True])
def test_launch_multirun(
    overrides, multirun_overrides, with_log_configuration, version_base
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
        with_log_configuration=with_log_configuration,
        multirun=True,
        **version_base,
    )
    assert isinstance(job, list) and len(job) == 1
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}

    if override_exists == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
def test_launch_with_multirun_overrides(version_base):
    cfg = builds(dict, a=1, b=1)
    multirun_overrides = ["hydra/sweeper=basic", "a=1,2"]
    with pytest.raises(ConfigCompositionException):
        launch(cfg, instantiate, overrides=multirun_overrides, **version_base)


###############################################
# Test local plugins work with hydra_zen.launch
###############################################


class LocalBasicSweeper(Sweeper):
    def setup(self, *, hydra_context, task_function, config):
        from hydra.core.plugins import Plugins

        self.hydra_context = hydra_context
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    def sweep(self, arguments):
        assert self.launcher is not None
        assert self.hydra_context is not None

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        override = parser.parse_overrides(arguments)[0]
        key = override.get_key_element()
        sweep = [f"{key}={val}" for val in override.sweep_string_iterator()]
        overrides = [[x] for x in sweep]

        returns = []
        for i, batch in enumerate(overrides):
            result = self.launcher.launch([batch], initial_job_idx=i)[0]
            returns.append(result)

        return [returns]


cs = ConfigStore.instance()
cs.store(group="hydra/sweeper", name="local_test", node=builds(LocalBasicSweeper))


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "plugin",
    [["hydra/sweeper=basic"], ["hydra/sweeper=local_test"]],
)
def test_launch_with_multirun_plugin(plugin, version_base):
    cfg = builds(dict, a=1, b=1)
    multirun_overrides = plugin + ["a=1,2"]
    job = launch(
        cfg, instantiate, overrides=multirun_overrides, multirun=True, **version_base
    )
    assert isinstance(job, list) and len(job) == 1 and len(job[0]) == 2
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}


@pytest.mark.filterwarnings(
    "ignore:Future Hydra versions will no longer change working directory"
)
@pytest.mark.parametrize("version_base", ["1.1", "1.2", None])
@pytest.mark.usefixtures("cleandir")
def test_version_base(version_base: Optional[str]):
    def task(cfg):
        (Path().cwd() / "foo.txt").touch()

    expected_dir = Path().cwd() if version_base != "1.1" else (Path().cwd() / "outputs")

    glob_pattern = "foo.txt" if version_base != "1.1" else "./**/foo.txt"

    assert len(list(expected_dir.glob(glob_pattern))) == 0

    launch(make_config(), task, version_base=version_base)
    assert len(list(expected_dir.glob(glob_pattern))) == 1, list(expected_dir.glob("*"))

    # ensure the file isn't found in the opposite location
    not_found = (
        Path().cwd().glob("foo.txt")
        if version_base == "1.1"
        else (Path().cwd() / "outputs").glob("**/foo.txt")
    )
    assert len(list(not_found)) == 0
