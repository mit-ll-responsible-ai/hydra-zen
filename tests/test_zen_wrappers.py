# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import string
from typing import Any, Callable, Dict, List, TypeVar, Union

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from omegaconf.errors import InterpolationKeyError
from typing_extensions import Protocol

from hydra_zen import builds, get_target, hydrated_dataclass, instantiate, just
from hydra_zen.structured_configs._implementations import is_builds
from hydra_zen.structured_configs._utils import is_interpolated_string
from hydra_zen.typing import Just, PartialBuilds
from hydra_zen.typing._implementations import InterpStr

T = TypeVar("T", bound=Callable)


class TrackedFunc(Protocol):
    tracked_id: int

    def __call__(self, obj: T) -> T:
        ...


def _coordinate_meta_fields_for_interpolation(wrappers, zen_meta):
    # Utility for testing
    #
    # Check if any of the wrappers are interpolation strings.
    # If so: attach corresponding meta-fields so that the
    # interpolated strings map to the named decorators
    if is_interpolated_string(wrappers):
        # change level of interpolation
        wrappers = wrappers.replace("..", ".")  # type: ignore
        dec_name: str = wrappers[3:-1]
        zen_meta[dec_name] = just(decorators_by_name[dec_name])
    elif isinstance(wrappers, list):
        for wrapper in wrappers:
            if is_interpolated_string(wrapper):
                if len(wrappers) == 1:
                    wrappers[0] = wrapper.replace("..", ".")
                dec_name = wrapper[4:-1]
                zen_meta[dec_name] = just(decorators_by_name[dec_name])
    return wrappers, zen_meta


def _resolve_wrappers(wrappers) -> List[TrackedFunc]:
    # Utility for testing
    if callable(wrappers) or isinstance(wrappers, str):
        wrappers = [wrappers]

    wrappers = [get_target(w) if is_builds(w) else w for w in wrappers]  # type: ignore
    wrappers = [
        decorators_by_name[w[2:-1].replace(".", "")] if isinstance(w, str) else w
        for w in wrappers
    ]
    return wrappers


def tracked_decorator(obj):
    if hasattr(obj, "num_decorated"):
        obj.num_decorated = obj.num_decorated + 1
    else:
        obj.num_decorated = 1
    return obj


# We will append the tracking-id of each wrapper function
# that is used.
TRACKED = []


def f1(obj):
    TRACKED.append(f1.tracked_id)
    return tracked_decorator(obj)


def f2(obj):
    TRACKED.append(f2.tracked_id)
    return tracked_decorator(obj)


def f3(obj):
    TRACKED.append(f3.tracked_id)
    return tracked_decorator(obj)


f1.tracked_id = 1
f2.tracked_id = 2
f3.tracked_id = 3

decorators_by_name = dict(f1=f1, f2=f2, f3=f3)


def target(*args, **kwargs):
    return args, kwargs


# prepare all variety of valid decorators to be tested
tracked_funcs = [f1, f2, f3]  # adds TrackedFunc
tracked_funcs.extend(just(f) for f in [f1, f2, f3])  # adds Just[TrackedFunc]
tracked_funcs.extend(builds(f, zen_partial=True) for f in [f1, f2, f3])
tracked_funcs.extend(["${..f1}", "${..f2}", "${..f3}"])

a_tracked_wrapper = st.sampled_from(tracked_funcs)


@settings(max_examples=500)  # ensures coverage of various branches
@given(
    wrappers=a_tracked_wrapper | st.lists(a_tracked_wrapper),
    args=st.lists(st.integers()),
    kwargs=st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1, max_size=1), st.integers()
    ),
    zen_partial=st.booleans(),
    zen_meta=st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1, max_size=1).map(lambda x: "_" + x),
        st.integers(),
        max_size=2,
    ),
)
def test_zen_wrappers_expected_behavior(
    wrappers: Union[  # type: ignore
        Union[TrackedFunc, Just[TrackedFunc], PartialBuilds[TrackedFunc], InterpStr],
        List[
            Union[TrackedFunc, Just[TrackedFunc], PartialBuilds[TrackedFunc], InterpStr]
        ],
    ],
    args: List[int],
    kwargs: Dict[str, int],
    zen_partial: bool,
    zen_meta: Dict[str, Any],
):
    """
    Tests:
    - wrappers as functions
    - wrappers as PartialBuilds
    - wrappers as Just
    - wrappers as interpolated strings
    - one or more wrappers
    - that each wrapper is called once, in order, from left to right
    - that each wrapper is passed the output of the previous wrapper
    - that the args and kwargs passed to the target are passed as-expected
    - that things interact as-expected with `zen_partial=True`
    - that things interact as-expected with `zen_meta`
    """
    TRACKED.clear()
    if hasattr(target, "num_decorated"):
        del target.num_decorated

    wrappers, zen_meta = _coordinate_meta_fields_for_interpolation(wrappers, zen_meta)  # type: ignore

    args = tuple(args)  # type: ignore
    conf = builds(
        target,
        *args,
        **kwargs,
        zen_wrappers=wrappers,
        zen_partial=zen_partial,
        zen_meta=zen_meta
    )
    instantiated = instantiate(conf)

    out_args, out_kwargs = instantiated() if zen_partial else instantiated  # type: ignore

    assert out_args == args
    assert out_kwargs == kwargs

    # ensure zen_meta works as-expected
    for meta_key, meta_val in zen_meta.items():
        assert getattr(conf, meta_key) == meta_val

    resolved_wrappers = _resolve_wrappers(wrappers)

    if wrappers:
        assert len(resolved_wrappers) == target.num_decorated
        assert TRACKED == [w.tracked_id for w in resolved_wrappers]
    else:
        assert not hasattr(target, "num_decorated")
        assert not TRACKED


def test_wrapper_for_hydrated_dataclass():
    TRACKED.clear()
    if hasattr(target, "num_decorated"):
        del target.num_decorated

    @hydrated_dataclass(target, zen_wrappers=f1)
    class A:
        pass

    instantiate(A)
    assert target.num_decorated == 1
    assert TRACKED == [f1.tracked_id]


class NotAWrapper:
    pass


@pytest.mark.parametrize(
    "bad_wrapper",
    [
        1,  # not callable,
        (None,),  # not callable in sequence
        (tracked_decorator, None),  # 1st ok, 2nd bad
    ],
)
def test_zen_wrappers_validation_during_builds(bad_wrapper):
    with pytest.raises(TypeError):
        builds(int, zen_wrappers=bad_wrapper)


@pytest.mark.parametrize(
    "bad_wrapper",
    [
        NotAWrapper,  # doesn't instantiate to a callable
        (NotAWrapper,),
        (tracked_decorator, NotAWrapper),
    ],
)
def test_zen_wrappers_validation_during_instantiation(bad_wrapper):
    conf = builds(int, zen_wrappers=bad_wrapper)
    with pytest.raises(TypeError):
        instantiate(conf)


@pytest.mark.parametrize(
    "bad_wrapper",
    ["${unresolved}", (tracked_decorator, "${unresolved}")],
)
def test_unresolved_interpolated_value_gets_caught(bad_wrapper):
    conf = builds(int, zen_wrappers=bad_wrapper)
    with pytest.raises(InterpolationKeyError):
        instantiate(conf)
