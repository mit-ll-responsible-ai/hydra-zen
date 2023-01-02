# Copyright (c) 2022 achusetts Institute of Technology
# SPDX-License-Identifier: MIT

import functools
from typing import Any

import hypothesis.strategies as st

from hydra_zen import to_yaml
from hydra_zen.structured_configs._utils import is_classmethod

valid_hydra_literals = st.sampled_from(
    [0, 1.0, True, "x", None, ["a", 1], {1: 1}, {"a": 1}]
)


def everything_except(*excluded_types: type):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def is_same(new, original):
    if isinstance(original, functools.partial):
        if not isinstance(new, functools.partial):
            return False
        return (
            new.args == original.args
            and new.keywords == original.keywords
            and new.func is original.func
        )
    if not is_classmethod(original):
        return new is original
    # objects like classmethods do not satisfy `x is x`
    return new == original


def sorted_yaml(conf: Any):
    return "\n".join(sorted(to_yaml(conf).splitlines()))
