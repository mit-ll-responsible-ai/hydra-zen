# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest
from omegaconf import DictConfig

from hydra_zen import ZenStore, builds, launch, make_config, to_yaml, zen


def relu():
    ...


def selu():
    ...


class Model:
    def __init__(self, activation_fn=relu):
        ...


def app(zen_cfg: DictConfig, model: Model) -> None:
    print(to_yaml(zen_cfg, resolve=True))


@pytest.mark.usefixtures("cleandir")
@pytest.mark.usefixtures("clean_store")
def test_merge_prevented_by_frozen_regression():
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/449

    store = ZenStore()
    Config = builds(
        app,
        populate_full_signature=True,
        hydra_defaults=[
            "_self_",
            {"model": "Model"},
            {"experiment": "selu"},
        ],
    )

    experiment_store = store(group="experiment", package="_global_")
    experiment_store(
        make_config(
            hydra_defaults=["_self_"], model=dict(activation_fn=selu), bases=(Config,)
        ),
        name="selu",
    )
    store(Model, group="model")
    store.add_to_hydra_store()
    launch(Config, zen(app), version_base="1.2")
