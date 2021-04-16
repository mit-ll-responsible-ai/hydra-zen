# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest
from hydra.core.config_store import ConfigStore
from hydra.errors import ConfigCompositionException

from hydra_zen import builds
from hydra_zen.experimental import hydra_launch
from hydra_zen.experimental._implementations import _load_config, _store_config


@pytest.mark.usefixtures("cleandir")
def test_hydra_launch_with_overrides_not_needed():
    # hydra config is loaded into task_cfg, so overrides is not used
    cn = _store_config(builds(dict, a=1, b=1))
    task_cfg = _load_config(cn)

    with pytest.raises(ValueError):
        hydra_launch(
            task_cfg, task_function=lambda _: print("hello"), overrides=["a=1,2"]
        )


@pytest.mark.usefixtures("cleandir")
def test_hydra_launch_with_multirun_in_overrides():
    # hydra config is not in task_cfg but multirun overrides must be in multirun_overrides
    task_cfg = builds(dict, a=1)

    with pytest.raises(ConfigCompositionException):
        hydra_launch(
            task_cfg, task_function=lambda _: print("hello"), overrides=["a=1,2"]
        )


@pytest.mark.usefixtures("cleandir")
def test_hydra_launch_multirun_with_default_values():
    # Define configs in the configstore
    b2_cfg = builds(dict, b=2)
    b4_cfg = builds(dict, b=4)

    cs = ConfigStore().instance()
    cs.store(group="a", name="b2", node=b2_cfg)
    cs.store(group="a", name="b4", node=b4_cfg)

    task_cfg = builds(dict, a=None)
    # verify it runs
    hydra_launch(
        task_cfg,
        task_function=lambda _: print("hello"),
        multirun_overrides=["+a=b2,b4"],
    )

    # need the + sign
    with pytest.raises(ConfigCompositionException):
        hydra_launch(
            task_cfg,
            task_function=lambda _: print("hello"),
            multirun_overrides=["a=b2,b4"],
        )

    # a default is set for "a"
    task_cfg = dict(f=builds(dict, a=b2_cfg))
    with pytest.raises(ConfigCompositionException):
        hydra_launch(
            task_cfg,
            task_function=lambda _: print("hello"),
            multirun_overrides=["a=b2,b4"],
        )
