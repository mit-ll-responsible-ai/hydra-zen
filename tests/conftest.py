# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import importlib
import logging
import os
import sys
import tempfile

import hypothesis.strategies as st
import pytest
from omegaconf import DictConfig, ListConfig

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []

OPTIONAL_TEST_DEPENDENCIES = (
    "numpy",
    "torch",
    "jax",
    "pytorch_lightning",
    "pydantic",
    "beartype",
)

for _module_name in OPTIONAL_TEST_DEPENDENCIES:
    try:
        importlib.import_module(_module_name)
    except ModuleNotFoundError:
        collect_ignore_glob.append(f"**/*{_module_name}*.py")

if sys.version_info < (3, 8):
    collect_ignore_glob.append("*py38*")

if sys.version_info < (3, 9):
    collect_ignore_glob.append("*py39*")

if sys.version_info > (3, 6):
    collect_ignore_glob.append("*py36*")

if sys.version_info < (3, 7):
    collect_ignore_glob.append("**/*test_sequence_coercion.py")


@pytest.fixture()
def cleandir():
    """Run function in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        os.chdir(tmpdirname)  # change cwd to the temp-directory
        yield tmpdirname  # yields control to the test to be run
        os.chdir(old_dir)
        logging.shutdown()


st.register_type_strategy(ListConfig, st.lists(st.integers()).map(ListConfig))
st.register_type_strategy(
    DictConfig, st.dictionaries(st.integers(), st.integers()).map(DictConfig)
)
