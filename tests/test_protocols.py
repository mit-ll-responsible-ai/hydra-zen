# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from functools import partial

import pytest

from hydra_zen import builds, just
from hydra_zen.typing import Builds, Just, PartialBuilds


@pytest.mark.parametrize(
    "fn, protocol",
    [
        (just, Just),
        (builds, Builds),
        (partial(builds, hydra_partial=True), PartialBuilds),
    ],
)
def test_runtime_checkability_of_protocols(fn, protocol):
    conf = fn(dict)()
    assert isinstance(conf, protocol)


# def test_build_p:pass
