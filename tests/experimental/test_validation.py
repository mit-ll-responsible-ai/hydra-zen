import pytest
from omegaconf import OmegaConf

from hydra_utils.experimental import hydra_launch
from hydra_utils.experimental._implementations import get_hydra_cfg


def test_hydra_launch_with_multiple_hydra_overrides_raises():

    task_cfg = OmegaConf.create(
        dict(hydra=get_hydra_cfg()),
    )

    with pytest.raises(ValueError):
        hydra_launch(
            task_cfg, task_function=lambda _: print("hello"), hydra_overrides=["a=1,2"]
        )
