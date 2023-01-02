# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import random

from hydra.core.config_store import ConfigStore

from hydra_zen import make_config, zen

cs = ConfigStore.instance()
cs.store(name="my_app", node=make_config("x", "y", z="${y}", seed=12))


@zen(pre_call=lambda cfg: random.seed(cfg.seed))
def f(x: int, y: int, z: int):
    pre_seeded = random.randint(0, 10)
    random.seed(12)
    seeded = random.randint(0, 10)
    assert pre_seeded == seeded
    return x + y + z


if __name__ == "__main__":
    f.hydra_main(config_name="my_app", config_path=None)
