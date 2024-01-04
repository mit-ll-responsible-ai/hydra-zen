# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from pytest import Config

import hydra_zen


def test_version():
    assert isinstance(hydra_zen.__version__, str)
    assert hydra_zen.__version__
    assert "unknown" not in hydra_zen.__version__


def test_xfail_strict(pytestconfig: Config):
    # Our test suite's xfail must be configured to strict mode
    # in order to ensure that contrapositive tests will actually
    # raise.
    assert pytestconfig.getini("xfail_strict") is True
