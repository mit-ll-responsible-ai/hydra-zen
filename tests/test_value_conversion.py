# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import string
from collections import Counter, deque
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Union

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, assume, given, settings
from omegaconf import DictConfig, ListConfig, OmegaConf

from hydra_zen import builds, get_target, instantiate, make_config, to_yaml
from hydra_zen._compatibility import ZEN_SUPPORTED_PRIMITIVES
from hydra_zen.structured_configs._value_conversion import ZEN_VALUE_CONVERSION


def test_supported_primitives_in_sync_with_value_conversion():
    assert len(set(ZEN_SUPPORTED_PRIMITIVES)) == len(ZEN_SUPPORTED_PRIMITIVES)
    assert set(ZEN_SUPPORTED_PRIMITIVES) == set(ZEN_VALUE_CONVERSION)


class Shake(Enum):
    VANILLA = 7
    CHOCOLATE = 4
    COOKIES = 9
    MINT = 3


# Needed for python 3.6
def is_ascii(x: str) -> bool:
    try:
        x.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


@pytest.mark.parametrize(
    "zen_supported_type",
    (
        # Hydra supported primitives
        int,
        float,
        bool,
        str,
        type(None),
        Shake,
        List[Union[int, List[int], Dict[int, int]]],
        Dict[int, Union[int, List[int], Dict[int, int]]],
        ListConfig,
        DictConfig,
        # hydra-zen supported primitives
        set,
        frozenset,
        FrozenSet[Union[int, complex]],
        Set[Union[int, complex, Path]],
        complex,
        Path,
        bytes,
        bytearray,
        deque,
        range,
        Counter,
    ),
)
@settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=(HealthCheck.data_too_large, HealthCheck.too_slow),
)
@given(data=st.data(), as_builds=st.booleans(), via_yaml=st.booleans())
def test_value_supported_via_config_maker_functions(
    zen_supported_type, data: st.DataObject, as_builds: bool, via_yaml: bool
):
    if zen_supported_type is str:
        value = data.draw(st.text(alphabet=string.ascii_letters))
    else:
        value = data.draw(st.from_type(zen_supported_type))

    Conf = (
        make_config(a=value, hydra_convert="all")
        if not as_builds
        else builds(dict, a=value, hydra_convert="all")
    )

    if via_yaml:
        if isinstance(value, Enum):
            # Default omegaconf support for enums doesn't roundtrip from yamls
            assume(False)

        Conf = OmegaConf.structured(to_yaml(Conf))

    conf = instantiate(Conf)

    if not isinstance(value, (ListConfig, DictConfig)):
        assert isinstance(conf["a"], type(value))

    if value == value:  # avoid nans
        if (
            not isinstance(value, range)
            and hasattr(value, "__iter__")
            and any(v != v for v in value)  # type: ignore
        ):  # avoid nans in collections
            pass
        else:
            assert conf["a"] == value


@given(x=st.lists(st.integers(-2, 2)))
def test_Counter_roundtrip(x):
    counter = Counter(x)
    BuildsConf = builds(dict, counter=counter)

    out_counter = instantiate(OmegaConf.structured(to_yaml(BuildsConf)))["counter"]

    assert counter == out_counter


def f1(*args, **kwargs):
    return args, kwargs


def f2(*args, **kwargs):
    return


@given(
    target=st.sampled_from([f1, f2]),
    args=st.lists(st.integers() | st.text(string.ascii_letters)).map(tuple),
    kwargs=st.dictionaries(
        keys=st.text("abcd", min_size=1),
        values=st.integers() | st.text(string.ascii_letters),
    ),
    as_builds=st.booleans(),
    via_yaml=st.booleans(),
)
def test_functools_partial_as_configured_value(
    target, args, kwargs, as_builds: bool, via_yaml: bool
):
    partiald_obj = partial(target, *args, **kwargs)
    Conf = (
        make_config(field=partiald_obj)
        if not as_builds
        else builds(dict, field=partiald_obj)
    )

    if via_yaml:
        Conf = OmegaConf.structured(to_yaml(Conf))

    out = instantiate(Conf)["field"]

    assert isinstance(out, partial)
    assert out.func is target
    assert out.args == args
    assert out.keywords == kwargs


def f3(z):
    return


def test_functools_partial_gets_validated():
    make_config(x=partial(f3, z=2))  # OK

    with pytest.raises(TypeError):
        # no param named `y`
        make_config(x=partial(f3, y=2))  # type: ignore


def f4(a, b=1, c="2"):
    return a, b, c


@settings(max_examples=500)
@given(
    partial_args=st.lists(st.integers(1, 4), max_size=3),
    partial_kwargs=st.dictionaries(
        keys=st.sampled_from("abc"), values=st.integers(-5, -2)
    ),
    args=st.lists(st.integers(10, 14), max_size=3),
    kwargs=st.dictionaries(keys=st.sampled_from("abc"), values=st.integers(-5, -2)),
)
def test_functools_partial_as_target(partial_args, partial_kwargs, args, kwargs):
    # Ensures that resolving a partial'd object behaves the exact same way as
    # configuring the object via `builds` and instantiating it.
    partiald_obj = partial(f4, *partial_args, **partial_kwargs)
    try:
        # might be under or over-specified
        out = partiald_obj(*args, **kwargs)
    except Exception as e:
        # `builds` should raise the same error, if over-specified
        # (under-specified configs are ok)
        if partial_args or args or "a" in partial_kwargs or "a" in kwargs:
            with pytest.raises(type(e)):
                builds(partiald_obj, *args, **kwargs)
    else:
        Conf = builds(partiald_obj, *args, **kwargs)
        assert out == instantiate(Conf)


def test_builds_handles_recursive_partial():
    # `functools.partial` automatically flattens itself:
    # partial(partial(dict, a=1) b=2) -> partial(dict, a=1, b=2)
    #
    # Thus `builds` should handle arbitrarily-nested partials "for free".
    # This test ensures this doesnt regress.
    partiald_obj = partial(partial(partial(f1, "a", a=1), "b"), b=2)
    Conf = builds(partiald_obj)

    assert get_target(Conf) is f1
    assert instantiate(Conf) == partiald_obj()
    assert partiald_obj() == (("a", "b"), dict(a=1, b=2))
