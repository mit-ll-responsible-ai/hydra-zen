from dataclasses import dataclass
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, instantiate


@dataclass
class TrackCall:
    num_calls: int = 0

    def __post_init__(self):
        self.funcs = []

    def __call__(self, fn) -> Any:
        self.num_calls += 1
        self.funcs.append(fn)
        return fn


@given(calls=st.lists(st.sampled_from(["no_wrap", "wrap", "error_wrap"])))
def test_instantiate_wrapper_restores_state(calls):
    wrapper_tracker = TrackCall()

    def error_wrap(x):
        raise ValueError("boom")

    cfg = builds(dict)
    for c in calls:
        pre_call_count = wrapper_tracker.num_calls
        if c == "no_wrap":
            instantiate(cfg)
            assert pre_call_count == wrapper_tracker.num_calls
        elif c == "error_wrap":

            # hydra module should be restored even if we hit an error
            with pytest.raises(ValueError):
                instantiate(cfg, _target_wrapper_=error_wrap)

            assert pre_call_count == wrapper_tracker.num_calls
        else:
            instantiate(cfg, _target_wrapper_=wrapper_tracker)
            assert pre_call_count + 1 == wrapper_tracker.num_calls


@given(...)
def test_wrapper_with_various_config_flavors(meta: bool, partial: bool):
    # test that wrapper is applied for partial instantiation and with zen-processing
    # features
    wrapper_tracker = TrackCall()
    kw = {}
    if meta:
        kw["zen_meta"] = {"a": 1}
    if partial:
        kw["zen_partial"] = True

    cfg = builds(dict, x=22, **kw)
    out = instantiate(cfg, _target_wrapper_=wrapper_tracker)
    assert wrapper_tracker.num_calls == 1
    assert wrapper_tracker.funcs[0] is dict
    if partial:
        out = out()  # type: ignore
    assert out == dict(x=22)


@given(...)
def test_recursive_wrappers(inner_meta: bool, outer_meta: bool):
    wrapper_tracker = TrackCall()
    inner_kw = {}
    if inner_meta:
        inner_kw["zen_meta"] = {"a": 1}
    outer_kw = {}
    if outer_meta:
        outer_kw["zen_meta"] = {"b": 2}
    cfg = builds(dict, x=builds(tuple, [1, 2, 3], **inner_kw), **outer_kw)
    out = instantiate(cfg, _target_wrapper_=wrapper_tracker)
    assert wrapper_tracker.num_calls == 2
    assert wrapper_tracker.funcs[0] is tuple
    assert wrapper_tracker.funcs[1] is dict
    assert out == dict(x=(1, 2, 3))
