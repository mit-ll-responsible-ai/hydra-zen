# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest
from hydra.core.config_store import ConfigStore
from hydra.errors import ConfigCompositionException

from hydra_zen import builds, instantiate
from hydra_zen.experimental import hydra_multirun, hydra_run
from hydra_zen.experimental._implementations import _load_config, _store_config


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
@pytest.mark.parametrize("hydra_overrides", [None, ["hydra.run.dir=test"]])
def test_hydra_run_with_hydra_in_config(overrides, hydra_overrides, expected):
    # validate hydra_launch executes properly if config contains
    # hydra configuration object
    cn = _store_config(builds(dict, a=1, b=1))
    task_cfg = _load_config(cn, overrides=hydra_overrides)
    assert "hydra" in task_cfg
    if hydra_overrides is not None:
        assert task_cfg.hydra.run.dir == "test"

    tf = lambda cfg: instantiate(cfg)
    # Provide user override
    task_cfg.b = 10

    # override works and user value is set
    job = hydra_run(task_cfg, task_function=tf, overrides=overrides)
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
@pytest.mark.parametrize("hydra_overrides", [None, ["hydra.sweep.dir=test"]])
def test_hydra_multirun_with_hydra_in_config(overrides, hydra_overrides, expected):
    # validate hydra_launch executes properly if config contains
    # hydra configuration object
    cn = _store_config(builds(dict, a=1, b=1))
    task_cfg = _load_config(cn, overrides=hydra_overrides)
    assert "hydra" in task_cfg
    if hydra_overrides is not None:
        assert task_cfg.hydra.sweep.dir == "test"

    tf = lambda cfg: instantiate(cfg)
    # Provide user override
    task_cfg.b = 10

    # override works and user value is set
    job = hydra_multirun(task_cfg, task_function=tf, overrides=overrides)
    for e, j, k in zip(expected, job[0], range(len(expected))):
        assert j.return_value == e
        assert j.working_dir == f"tested/{k}"


@pytest.mark.usefixtures("cleandir")
def test_hydra_run_with_multirun_in_overrides():
    # hydra config is not in task_cfg but multirun overrides must be in multirun_overrides
    task_cfg = builds(dict, a=1)

    with pytest.raises(ConfigCompositionException):
        hydra_run(task_cfg, task_function=lambda _: print("hello"), overrides=["a=1,2"])


@pytest.mark.usefixtures("cleandir")
def test_hydra_multirun_with_default_values():
    # Define configs in the configstore
    b2_cfg = builds(dict, b=2)
    b4_cfg = builds(dict, b=4)

    cs = ConfigStore().instance()
    cs.store(group="a", name="b2", node=b2_cfg)
    cs.store(group="a", name="b4", node=b4_cfg)

    task_cfg = builds(dict, a=None)
    # verify it runs
    hydra_multirun(
        task_cfg,
        task_function=lambda _: print("hello"),
        overrides=["+a=b2,b4"],
    )

    # need the + sign
    with pytest.raises(ConfigCompositionException):
        hydra_multirun(
            task_cfg,
            task_function=lambda _: print("hello"),
            overrides=["a=b2,b4"],
        )

    # a default is set for "a"
    task_cfg = dict(f=builds(dict, a=b2_cfg))
    with pytest.raises(ConfigCompositionException):
        hydra_multirun(
            task_cfg,
            task_function=lambda _: print("hello"),
            overrides=["a=b2,b4"],
        )
