# Copyright (c) 2021 Massachusetts Institute of Technology

import hypothesis.strategies as st

valid_hydra_literals = st.sampled_from([0, 1.0, True, "x", ["a", 1], {1: 1}, {"a": 1}])
