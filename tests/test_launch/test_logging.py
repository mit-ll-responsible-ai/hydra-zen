# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import pytest
from hydra.core.utils import JobReturn

from hydra_zen import builds, instantiate, launch

log = logging.getLogger(__name__)


def task(cfg):
    instantiate(cfg)
    log.info(f"message: {cfg['message']}")


@pytest.mark.usefixtures("cleandir")
def test_consecutive_logs():
    job1 = launch(
        builds(dict, message="1"), task_function=task, overrides=["hydra.run.dir=test"]
    )
    job2 = launch(
        builds(dict, message="2"), task_function=task, overrides=["hydra.run.dir=test"]
    )

    assert isinstance(job1, JobReturn) and job1.working_dir is not None
    assert isinstance(job2, JobReturn) and job2.working_dir is not None
    assert job1.working_dir == "test"
    assert job1.working_dir == job2.working_dir

    with open(Path(job1.working_dir) / f"{job1.hydra_cfg.hydra.job.name}.log") as f:
        line1, line2 = f.readlines()
    assert "message: 1" in line1
    assert "message: 2" in line2


@pytest.mark.usefixtures("cleandir")
def test_seperate_logs():
    job1 = launch(
        builds(dict, message="1"), task_function=task, overrides=["hydra.run.dir=test1"]
    )
    job2 = launch(
        builds(dict, message="2"), task_function=task, overrides=["hydra.run.dir=test2"]
    )

    assert isinstance(job1, JobReturn) and job1.working_dir is not None
    assert isinstance(job2, JobReturn) and job2.working_dir is not None

    for n, job in enumerate([job1, job2]):
        with open(Path(job.working_dir) / f"{job.hydra_cfg.hydra.job.name}.log") as f:
            (line,) = f.readlines()
        assert f"message: {n + 1}" in line
