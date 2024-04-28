# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import pickle
import string
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Set, Union

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, assume, given, settings
from omegaconf import DictConfig, ListConfig, OmegaConf, ValidationError

from hydra_zen import (
    builds,
    get_target,
    instantiate,
    make_config,
    make_custom_builds_fn,
    mutable_value,
    to_yaml,
)
from hydra_zen._compatibility import ZEN_SUPPORTED_PRIMITIVES
from hydra_zen.structured_configs._implementations import ZEN_VALUE_CONVERSION
from hydra_zen.typing import Partial
from tests import is_same


def test_supported_primitives_in_sync_with_value_conversion():
    assert len(set(ZEN_SUPPORTED_PRIMITIVES)) == len(ZEN_SUPPORTED_PRIMITIVES)
    assert set(ZEN_SUPPORTED_PRIMITIVES) == set(ZEN_VALUE_CONVERSION)


class Shake(Enum):
    VANILLA = 7
    CHOCOLATE = 4
    COOKIES = 9
    MINT = 3


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
        timedelta,
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


pik_blds = make_custom_builds_fn(
    zen_dataclass={"module": "tests.test_value_conversion"}
)

Bint = pik_blds(dict, x=1, zen_dataclass={"cls_name": "Bint"})
Bfloat = pik_blds(dict, x=1.0, zen_dataclass={"cls_name": "Bfloat"})
Bbool = pik_blds(dict, x=True, zen_dataclass={"cls_name": "Bbool"})
Bnone = pik_blds(dict, x=None, zen_dataclass={"cls_name": "Bnone"})
Benum = pik_blds(dict, x=Shake(3), zen_dataclass={"cls_name": "Benum"})
Blist = pik_blds(dict, x=[1, {"a": 2}], zen_dataclass={"cls_name": "Blist"})
Blistconfig = pik_blds(
    dict, x=ListConfig([1, {"a": 2}]), zen_dataclass={"cls_name": "Blistconfig"}
)
Bset = pik_blds(dict, x=set([1, 2]), zen_dataclass={"cls_name": "Bset"})
Bfrozenset = pik_blds(
    dict, x=frozenset([1j, 2j]), zen_dataclass={"cls_name": "Bfrozenset"}
)
Bcomplex = pik_blds(dict, x=1j, zen_dataclass={"cls_name": "Bcomplex"})
Bpath = pik_blds(dict, x=Path.cwd(), zen_dataclass={"cls_name": "Bpath"})
Bbytes = pik_blds(dict, x=b"123", zen_dataclass={"cls_name": "Bbytes"})
Bbytearray = pik_blds(
    dict, x=bytearray([1, 2]), zen_dataclass={"cls_name": "Bbytearray"}
)
Bdeque = pik_blds(dict, x=deque([1j, 2]), zen_dataclass={"cls_name": "Bdeque"})
Brange = pik_blds(dict, x=range(1, 10, 3), zen_dataclass={"cls_name": "Brange"})
Brange2 = pik_blds(dict, x=range(2), zen_dataclass={"cls_name": "Brange2"})
Bcounter = pik_blds(dict, x=Counter("apple"), zen_dataclass={"cls_name": "Bcounter"})
x = defaultdict(list)
x.update({1: [1, 2]})
Bdefaultdict = pik_blds(dict, x=x, zen_dataclass={"cls_name": "Bcounter"})


@pytest.mark.parametrize(
    "Config",
    [
        Bint,
        Bfloat,
        Bbool,
        Bnone,
        Benum,
        Blist,
        Blistconfig,
        Bset,
        Bfrozenset,
        Bcomplex,
        Bpath,
        Bbytes,
        Bbytearray,
        Bdeque,
        Brange,
        Brange2,
        Bcounter,
        Bdefaultdict,
    ],
)
def test_pickle_compatibility(Config):
    assert pickle.loads(pickle.dumps(Config)) is Config
    assert pickle.loads(pickle.dumps(Config())) == Config()


@pytest.mark.parametrize(
    "Config",
    [
        Bint,
        Bfloat,
        Bbool,
        Bnone,
        Benum,
        Blistconfig,
        Bset,
        Bfrozenset,
        Bcomplex,
        Bpath,
        Bbytes,
        Bbytearray,
        Bdeque,
        Brange,
        Brange2,
        Bcounter,
        Bdefaultdict,
    ],
)
def test_unsafe_hash_default(Config):
    assert Config().x.__hash__ is not None


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


@given(type_=st.sampled_from(list(ZEN_SUPPORTED_PRIMITIVES)), data=st.data())
def test_zen_supported_primitives_arent_supported_by_hydra(type_, data: st.DataObject):
    try:
        value = data.draw(st.from_type(type_))
    except (TypeError, NotImplementedError):
        assume(False)
        return

    @dataclass
    class C:
        x: type_ = mutable_value(value) if value.__hash__ is None else value

    with pytest.raises((ValidationError, AssertionError)):
        Conf = OmegaConf.create(C)  # type: ignore
        assert isinstance(Conf.x, type_)


@dataclass
class A_builds_populate_sig_with_default_factory:
    z: Any
    x_list: List[int] = field(default_factory=lambda: list([1, 0, 1, 0, 1]))
    x_dict: Dict[str, int] = field(default_factory=lambda: dict({"K_DEFAULT": 10101}))
    y: bool = False


@given(
    via_yaml=st.booleans(),
    list_=st.none() | st.lists(st.integers()),
    # TODO: generalize to st.dictionaries(st.sampled_from("abcd"), st.integers())
    #       once https://github.com/facebookresearch/hydra/issues/2350 is resolved
    dict_=st.none(),
    kwargs_via_builds=st.booleans(),
)
def test_builds_populate_sig_with_default_factory(
    via_yaml: bool, list_, dict_, kwargs_via_builds
):
    A = A_builds_populate_sig_with_default_factory
    kwargs = {}
    if list_ is not None:
        kwargs["x_list"] = list_

    if dict_ is not None:
        kwargs["x_dict"] = dict_

    Conf = (
        builds(A, **kwargs, populate_full_signature=True)
        if kwargs_via_builds
        else builds(A, populate_full_signature=True)
    )

    assert inspect.signature(Conf).parameters == inspect.signature(A).parameters

    if via_yaml:
        Conf = OmegaConf.structured(to_yaml(Conf))
    a_expected = A(z=1, **kwargs)

    a = (
        instantiate(Conf, z=1)
        if kwargs_via_builds
        else instantiate(Conf, **kwargs, z=1)
    )

    assert isinstance(a, A)
    assert a.x_list == a_expected.x_list
    assert a.x_dict == a_expected.x_dict
    assert a.y == a_expected.y
    assert a.z == a_expected.z


@dataclass
class A_auto_config_for_dataclass_fields:
    complex_factory: Any = mutable_value(1 + 2j)
    complex_: complex = 2 + 4j
    list_of_stuff: List[Any] = field(
        default_factory=lambda: list([1 + 2j, Path.home()])
    )
    fn_factory: Callable[[Iterable[int]], int] = field(default_factory=lambda: sum)
    fn: Callable[[Iterable[int]], int] = sum
    partial_factory: Partial[int] = field(default_factory=lambda: partial(sum))
    partial_: Partial[int] = partial(sum)


def test_auto_config_for_dataclass_fields():
    A = A_auto_config_for_dataclass_fields
    Conf = builds(A, populate_full_signature=True)
    actual = instantiate(Conf)
    expected = A()
    assert isinstance(actual, A)
    assert actual.complex_factory == expected.complex_factory
    assert actual.complex_ == expected.complex_
    assert actual.list_of_stuff == expected.list_of_stuff
    assert is_same(actual.fn_factory, expected.fn_factory)
    assert is_same(actual.fn, expected.fn)
    assert is_same(actual.partial_factory, expected.partial_factory)
    assert is_same(actual.partial_, expected.partial_)


def identity_with_dict_default(x={"a": 1}):
    return x


@pytest.mark.xfail
def test_known_failcase_hydra_2350():
    # https://github.com/facebookresearch/hydra/issues/2350
    # Overriding a default-value dictionary via instantiate interface results
    # in merging of the default-dictionary with the override value
    Conf = builds(identity_with_dict_default, populate_full_signature=True)
    actual = instantiate(Conf, x={"b": 2})
    expected = {"b": 2}
    assert actual == expected, actual


def test_default_dict():
    x = defaultdict(list)
    x.update({1: [1 + 2j, 2]})
    Conf = builds(dict, x=x)
    actual = instantiate(Conf)["x"]
    assert actual == x
    assert isinstance(actual, defaultdict)
    assert actual.default_factory is list
