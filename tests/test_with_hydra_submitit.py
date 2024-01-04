# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import pickle
import sys

import pytest


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Submitit doesn't work on Windows",
)
@pytest.mark.usefixtures("cleandir")
def test_pickling_with_hydra_main():
    """This test uses hydra-submitit-launcher because
    submitit uses cloudpickle to pickle the task function
    and execute a job from the pickled task function."""
    import subprocess
    from pathlib import Path

    path = (Path(__file__).parent / "example_app" / "dummy_zen_main.py").absolute()
    assert not (Path.cwd() / "multirun").is_dir()
    subprocess.run(
        ["python", path, "x=1", "y=2", "hydra/launcher=submitit_local", "--multirun"]
    ).check_returncode()
    assert (Path.cwd() / "multirun").is_dir()

    multirun_files = list(Path.cwd().glob("**/multirun.yaml"))
    assert len(multirun_files) == 1
    multirun_file = multirun_files[0]
    assert (multirun_file.parent / ".submitit").is_dir()

    # load the results saved by submitit
    pkls = list((multirun_file.parent / ".submitit").glob("**/*_result.pkl"))
    assert len(pkls) == 1
    with open(pkls[0], "rb") as f:
        result = pickle.load(f)

    # assert the result is correct and the task function executed
    assert result[0] == "success"
    assert result[1].return_value == 5
