# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st

valid_hydra_literals = st.sampled_from(
    [0, 1.0, True, "x", None, ["a", 1], {1: 1}, {"a": 1}]
)


def everything_except(*excluded_types: type):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )
