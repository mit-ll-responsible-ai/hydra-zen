# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from datetime import datetime

from hydra.conf import HydraConf, JobConf

from hydra_zen import store, zen

store(HydraConf(job=JobConf(chdir=True)))


@store
def task():
    # Used to test that configuring Hydra to change working dir to time-stamped
    # output dir works as-expected
    from pathlib import Path

    path = Path.cwd()
    assert path.parent.name == datetime.today().strftime("%Y-%m-%d")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(task).hydra_main(config_name="task", config_path=None)
