# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect

import hypothesis.strategies as st
import pytest
import pytorch_lightning as pl
from hypothesis import assume, given
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, just, to_yaml
from tests import check_identity

pl_objects = [
    pl.Trainer,
    pl.LightningDataModule,
    pl.LightningDataModule.from_datasets,
    pl.LightningModule,
    pl.accelerators.Accelerator,
    pl.callbacks.Callback,
    pl.callbacks.GPUStatsMonitor,
    pl.core.decorators.auto_move_data,
    pl.callbacks.early_stopping.EarlyStopping,
    pl.loggers.base.LightningLoggerBase,
    pl.core.hooks.CheckpointHooks,
    pl.callbacks.gradient_accumulation_scheduler.GradientAccumulationScheduler,
    pl.callbacks.lr_monitor.LearningRateMonitor,
    pl.callbacks.model_checkpoint.ModelCheckpoint,
    pl.loggers.comet.CometLogger,
    pl.plugins.training_type.TrainingTypePlugin,
    pl.plugins.training_type.DataParallelPlugin,
    pl.plugins.precision.PrecisionPlugin,
]


@pytest.mark.parametrize("obj", pl_objects)
def test_just_roundtrip(obj):
    assert check_identity(instantiate(just(obj)), obj)


@pytest.mark.parametrize("target", pl_objects)
@given(partial=st.booleans(), full_sig=st.booleans())
def test_fuzz_build_validation_against_a_bunch_of_common_objects(
    target, partial: bool, full_sig: bool
):
    doesnt_have_sig = False
    try:
        inspect.signature(target)
    except ValueError:
        doesnt_have_sig = True

    if doesnt_have_sig and full_sig:
        assume(False)
    conf = builds(target, zen_partial=partial, populate_full_signature=full_sig)

    OmegaConf.create(to_yaml(conf))  # ensure serializable

    if partial:
        instantiate(conf)  # ensure instantiable
