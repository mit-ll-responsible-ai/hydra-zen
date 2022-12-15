# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import NamedTuple

import pytest
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from hydra_zen import builds, instantiate, launch


class Tracker(NamedTuple):
    job_start: bool = False
    job_end: bool = False
    run_start: bool = False
    run_end: bool = False
    multirun_start: bool = False
    multirun_end: bool = False


class CustomCallback(Callback):
    JOB_START_CALLED = False
    JOB_END_CALLED = False

    RUN_START_CALLED = False
    RUN_END_CALLED = False

    MULTIRUN_START_CALLED = False
    MULTIRUN_END_CALLED = False

    def __init__(self, callback_name):
        self.name = callback_name

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        CustomCallback.JOB_START_CALLED = True

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs) -> None:
        CustomCallback.JOB_END_CALLED = True

    def on_run_start(self, config: DictConfig, **kwargs) -> None:
        CustomCallback.RUN_START_CALLED = True

    def on_run_end(self, config: DictConfig, **kwargs) -> None:
        CustomCallback.RUN_END_CALLED = True

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        CustomCallback.MULTIRUN_START_CALLED = True

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        CustomCallback.MULTIRUN_END_CALLED = True


cs = ConfigStore.instance()
cs.store(
    group="hydra/callbacks",
    name="test_callback",
    node=dict(test_callback=builds(CustomCallback, callback_name="test")),
)


def tracker(x=CustomCallback):
    # this will get called after the job and run have started
    # but before they end
    return Tracker(
        job_start=x.JOB_START_CALLED,
        job_end=x.JOB_END_CALLED,
        run_start=x.RUN_START_CALLED,
        run_end=x.RUN_END_CALLED,
        multirun_start=x.MULTIRUN_START_CALLED,
        multirun_end=x.MULTIRUN_END_CALLED,
    )


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("multirun", [False, True])
def test_hydra_run_with_callback(multirun, version_base: dict):
    # Tests that callback methods are called during appropriate
    # stages
    try:
        cfg = builds(tracker)

        assert not any(tracker())  # ensures all flags are false

        job = launch(
            cfg,
            task_function=instantiate,
            overrides=["hydra/callbacks=test_callback"],
            multirun=multirun,
            **version_base,
        )

        if multirun:
            job = job[0][0]

        tracked_mid_run: Tracker = job.return_value
        assert tracked_mid_run.job_start is True
        assert tracked_mid_run.run_start is not multirun
        assert tracked_mid_run.multirun_start is multirun

        assert tracked_mid_run.job_end is False
        assert tracked_mid_run.run_end is False
        assert tracked_mid_run.multirun_end is False

        assert CustomCallback.JOB_END_CALLED is True
        assert CustomCallback.RUN_END_CALLED is not multirun
        assert CustomCallback.MULTIRUN_END_CALLED is multirun

    finally:
        CustomCallback.JOB_START_CALLED = False
        CustomCallback.JOB_END_CALLED = False

        CustomCallback.RUN_START_CALLED = False
        CustomCallback.RUN_END_CALLED = False

        CustomCallback.MULTIRUN_START_CALLED = False
        CustomCallback.MULTIRUN_END_CALLED = False
