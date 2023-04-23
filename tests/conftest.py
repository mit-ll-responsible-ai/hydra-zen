# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import logging
import os
import sys
import tempfile
from copy import deepcopy
from typing import Dict, Iterable, Optional

import hypothesis.strategies as st
import pkg_resources
import pytest
from hydra.core.config_store import ConfigStore
from hypothesis import settings
from omegaconf import DictConfig, ListConfig

from hydra_zen import store
from hydra_zen._compatibility import HYDRA_VERSION

_store = ConfigStore.instance()

# Skip collection of tests that don't work on the current version of Python.
collect_ignore_glob = []

OPTIONAL_TEST_DEPENDENCIES = (
    "numpy",
    "torch",
    "jax",
    "pytorch_lightning",
    "pydantic",
    "beartype",
    "submitit",
)

_installed = {pkg.key for pkg in pkg_resources.working_set}

for _module_name in OPTIONAL_TEST_DEPENDENCIES:
    if _module_name not in _installed:
        collect_ignore_glob.append(f"*{_module_name}*.py")

if sys.version_info > (3, 6):
    collect_ignore_glob.append("*py36*")

if sys.version_info < (3, 7):
    collect_ignore_glob.append("**/*test_sequence_coercion.py")

if sys.version_info < (3, 8):
    collect_ignore_glob.append("*py38*")

if sys.version_info < (3, 9):
    collect_ignore_glob.append("*py39*")

if sys.version_info < (3, 10):
    collect_ignore_glob.append("*py310*")


@pytest.fixture()
def cleandir() -> Iterable[str]:
    """Run function in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        os.chdir(tmpdirname)  # change cwd to the temp-directory
        yield tmpdirname  # yields control to the test to be run
        os.chdir(old_dir)
        logging.shutdown()


@pytest.fixture()
def clean_store() -> Iterable[dict]:
    """Provides access to configstore repo and restores state after test"""
    prev_state = deepcopy(_store.repo)
    zen_prev_state = (store._internal_repo.copy(), store._queue.copy())
    yield _store.repo
    _store.repo = prev_state
    int_repo, queue = zen_prev_state
    store._internal_repo = int_repo
    store._queue = queue


@pytest.fixture()
def version_base() -> Dict[str, Optional[str]]:
    """Return version_base according to local version, or empty dict for versions
    preceding version_base"""
    return (
        {"version_base": ".".join(str(i) for i in HYDRA_VERSION)}
        if HYDRA_VERSION >= (1, 2, 0)
        else {}
    )


pytest_plugins = "pytester"

st.register_type_strategy(ListConfig, st.lists(st.integers()).map(ListConfig))
st.register_type_strategy(
    DictConfig, st.dictionaries(st.integers(), st.integers()).map(DictConfig)
)


settings.register_profile("ci", deadline=None)

if bool(os.environ.get("CI")):
    settings.load_profile("ci")
