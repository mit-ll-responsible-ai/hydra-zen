# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import os

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.errors import ConfigCompositionException

from hydra_zen import builds, instantiate, launch
from hydra_zen._launch import _store_config


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides, expected",
    [
        [["a=2", "hydra.run.dir=tested"], dict(a=2, b=10)],
        [
            ["a=2", "b=100", "hydra.run.dir=tested"],
            dict(a=2, b=100),
        ],
    ],
)
@pytest.mark.parametrize("hydra_overrides", [[], ["hydra.run.dir=test"]])
def test_launch_with_hydra_in_config(
    overrides, hydra_overrides, expected, version_base: dict
):
    # validate hydra_launch executes properly if config contains
    # hydra configuration object
    cn = _store_config(builds(dict, a=1, b=1))
    with initialize(config_path=None, **version_base):
        task_cfg = compose(
            config_name=cn, overrides=hydra_overrides, return_hydra_config=True
        )

    assert "hydra" in task_cfg
    if len(hydra_overrides) > 0:
        assert task_cfg.hydra.run.dir == "test"

    # Provide user override
    task_cfg.b = 10

    if version_base:
        overrides.append("hydra.job.chdir=True")
    # override works and user value is set
    job = launch(
        task_cfg,
        task_function=instantiate,
        overrides=overrides,
        **version_base,
    )
    assert job.return_value == expected
    assert job.working_dir == "tested"


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "overrides, expected",
    [
        [
            ["a=2,3", "hydra.sweep.dir=tested"],
            [dict(a=2, b=10), dict(a=3, b=10)],
        ],
        [
            ["b=12", "a=2,3", "hydra.sweep.dir=tested"],
            [dict(a=2, b=12), dict(a=3, b=12)],
        ],
    ],
)
@pytest.mark.parametrize("hydra_overrides", [[], ["hydra.sweep.dir=test"]])
def test_launch_with_multirun_with_hydra_in_config(
    overrides,
    hydra_overrides,
    expected,
    version_base: dict,
):
    # validate hydra_launch executes properly if config contains
    # hydra configuration object
    cn = _store_config(builds(dict, a=1, b=1))
    with initialize(config_path=None, **version_base):
        task_cfg = compose(
            config_name=cn,
            overrides=hydra_overrides,
            return_hydra_config=True,
        )

    assert "hydra" in task_cfg
    if len(hydra_overrides) > 0:
        assert task_cfg.hydra.sweep.dir == "test"

    # Provide user override
    task_cfg.b = 10

    if version_base:
        overrides.append("hydra.job.chdir=True")

    # override works and user value is set
    job = launch(
        task_cfg,
        task_function=instantiate,
        overrides=overrides,
        multirun=True,
        **version_base,
    )
    for e, j, k in zip(expected, job[0], range(len(expected))):
        assert j.return_value == e
        assert j.working_dir == f"tested{os.path.sep}{k}"


@pytest.mark.usefixtures("cleandir")
def test_launch_with_multirun_in_overrides(version_base):
    # hydra config is not in task_cfg but multirun overrides must be in multirun_overrides
    task_cfg = builds(dict, a=1)

    with pytest.raises(ConfigCompositionException):
        launch(
            task_cfg,
            task_function=lambda _: print("hello"),
            overrides=["a=1,2"],
            **version_base,
        )


@pytest.mark.usefixtures("cleandir")
def test_launch_with_multirun_with_default_values(version_base):
    # Define configs in the configstore
    b2_cfg = builds(dict, b=2)
    b4_cfg = builds(dict, b=4)

    cs = ConfigStore().instance()
    cs.store(group="a", name="b2", node=b2_cfg)
    cs.store(group="a", name="b4", node=b4_cfg)

    task_cfg = builds(dict, a=None)
    # verify it runs
    launch(
        task_cfg,
        task_function=lambda _: print("hello"),
        overrides=["+a=b2,b4"],
        multirun=True,
        **version_base,
    )

    # need the + sign
    with pytest.raises(ConfigCompositionException):
        launch(
            task_cfg,
            task_function=lambda _: print("hello"),
            overrides=["a=b2,b4"],
            multirun=True,
            **version_base,
        )

    # a default is set for "a"
    task_cfg = dict(f=builds(dict, a=b2_cfg))
    with pytest.raises(ConfigCompositionException):
        launch(
            task_cfg,
            task_function=lambda _: print("hello"),
            overrides=["a=b2,b4"],
            multirun=True,
            **version_base,
        )
