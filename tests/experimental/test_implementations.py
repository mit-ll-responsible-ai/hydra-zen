# Copyright (c) 2021 Massachusetts Institute of Technology
from pathlib import Path

import pytest
from omegaconf.omegaconf import OmegaConf

from hydra_utils import builds, instantiate
from hydra_utils.experimental import hydra_launch


def f(a: int = 1, b: int = 2):
    return dict(a=a, b=b)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "hydra_overrides",
    [[], ["hydra.run.dir=test_hydra_overrided"]],
)
@pytest.mark.parametrize("as_dataclass", [True, False])
def test_hydra_launch_job(hydra_overrides, as_dataclass):
    cfg = builds(f, a=1, b=1)
    if not as_dataclass:
        cfg = OmegaConf.create(cfg)

    job = hydra_launch(
        cfg,
        task_function=lambda x: instantiate(x),
        hydra_overrides=hydra_overrides,
    )
    assert job.return_value == {"a": 1, "b": 1}

    if len(hydra_overrides) == 1:
        assert Path("test_hydra_overrided").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "hydra_overrides",
    [[], ["hydra.sweep.dir=test_hydra_overrided"]],
)
@pytest.mark.parametrize("as_dataclass", [True, False])
@pytest.mark.parametrize("use_default_dir", [True, False])
def test_hydra_launch_multirun(hydra_overrides, as_dataclass, use_default_dir: bool):
    cfg = builds(f, a=1, b=1)
    if not as_dataclass:
        cfg = OmegaConf.create(cfg)

    additl_kwargs = {} if use_default_dir else dict(config_dir=Path.cwd())
    job = hydra_launch(
        cfg,
        task_function=lambda x: instantiate(x),
        multirun_overrides=["a=1,2"],
        hydra_overrides=hydra_overrides,
        **additl_kwargs,
    )
    for i, j in enumerate(job[0]):
        assert j.return_value == {"a": i + 1, "b": 1}

    if len(hydra_overrides) == 1:
        assert Path("test_hydra_overrided").exists()
