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
    Conf = fn(dict)
    assert isinstance(Conf, protocol)

    conf = Conf()
    assert isinstance(conf, protocol)


def test_Builds_is_not_PartialBuilds():
    Conf = builds(dict)
    assert not isinstance(Conf, PartialBuilds)

    PConf = builds(dict, hydra_partial=True)
    assert isinstance(PConf, Builds)
