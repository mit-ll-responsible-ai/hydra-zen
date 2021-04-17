# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile

import pytest

try:
    import numpy
except ImportError:
    pass

try:
    import torch
except ImportError:
    pass

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []

if sys.version_info < (3, 8):
    collect_ignore_glob.append("*py38*")

OPTIONAL_TEST_DEPENDENCIES = (
    "numpy",
    "torch",
    "jax"
)

for module in OPTIONAL_TEST_DEPENDENCIES:
    if module not in sys.modules:
        collect_ignore_glob.append(f"*{module}*")


@pytest.fixture()
def cleandir():
    """Run function in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        os.chdir(tmpdirname)  # change cwd to the temp-directory
        yield tmpdirname  # yields control to the test to be run
        os.chdir(old_dir)
