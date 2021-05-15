# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import pytest

from hydra_zen import builds, instantiate
from hydra_zen.experimental import hydra_run

log = logging.getLogger(__name__)


def task(cfg):
    instantiate(cfg)
    log.info(f"message: {cfg['message']}")


@pytest.mark.usefixtures("cleandir")
def test_consecutive_logs():
    job1 = hydra_run(builds(dict, message="1"), task_function=task)
    job2 = hydra_run(builds(dict, message="2"), task_function=task)

    if job1.working_dir == job2.working_dir:
        with open(Path(job1.working_dir) / "hydra_run.log", "r") as f:
            line1, line2 = f.readlines()
        assert "message: 1" in line1
        assert "message: 2" in line2

    else:
        for n, job in enumerate([job1, job2]):
            with open(Path(job.working_dir) / "hydra_run.log", "r") as f:
                (line,) = f.readlines()
            assert f"message: {n + 1}" in line
