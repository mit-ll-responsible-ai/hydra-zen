# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from pickle import PicklingError, dumps, loads
from typing import Optional

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import (
    builds,
    hydrated_dataclass,
    instantiate,
    just,
    make_config,
    make_custom_builds_fn,
    to_yaml,
)
from hydra_zen.errors import HydraZenDeprecationWarning


def f_three_vars(x, y, z):
    return x, y, z


def f(x, y, z: int = 3):
    return x, y, z


@given(full_sig=st.booleans(), partial=st.none() | st.booleans())
def test_builds_produces_dataclass(full_sig: bool, partial: bool):
    if full_sig and not partial:
        Builds_f = builds(f, populate_full_signature=full_sig, zen_partial=partial)
    else:
        Builds_f = builds(
            f, x=None, y=None, zen_partial=partial, populate_full_signature=full_sig
        )
    assert is_dataclass(Builds_f)
    out = Builds_f(x=1.0, y=-1.0)
    assert out.x == 1.0
    assert out.y == -1.0

    if full_sig:
        assert out.z == 3


def f_2(x, y, z):
    pass


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False, None])
def test_chain_builds_of_targets_with_common_interfaces(full_sig, partial: bool):
    # Note that conf_1 and conf_2 target `f` whereas conf_3 targets `f_three_vars`,
    # which have identical interfaces.
    conf_1 = builds(f_2, x=1)
    conf_2 = builds(f_2, y=2, builds_bases=(conf_1,))
    conf_3 = builds(
        f_three_vars,
        z=3,
        zen_partial=partial,
        populate_full_signature=full_sig,
        builds_bases=(conf_2,),
    )

    # checks subclass relationships
    assert conf_3.__mro__[:3] == (conf_3, conf_2, conf_1)

    out = instantiate(conf_3)
    if partial:
        # resolve partial
        out = out()  # type: ignore

    assert out == (1, 2, 3)


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False, None])
def test_pos_args_with_inheritance(full_sig, partial: bool):
    conf_1 = builds(f_three_vars, 1, 2)
    conf_2 = builds(
        f_three_vars,
        z=3,
        zen_partial=partial,
        populate_full_signature=full_sig,
        builds_bases=(conf_1,),
    )

    # checks subclass relationships
    assert conf_2.__mro__[:2] == (conf_2, conf_1)

    out = instantiate(conf_2)
    if partial:
        # resolve partial
        out = out()  # type: ignore

    assert out == (1, 2, 3)


def f_3(x):
    pass


@hydrated_dataclass(dict, frozen=True)
class FrozenHydrated:
    x: int = 2


@pytest.mark.filterwarnings("ignore:Specifying")
@pytest.mark.parametrize(
    "fn",
    [
        lambda: builds(dict, x=2, zen_dataclass={"frozen": True})(),
        lambda: builds(dict, x=2, frozen=True)(),
        lambda: make_custom_builds_fn(zen_dataclass={"frozen": True})(dict, x=2)(),
        lambda: make_custom_builds_fn(frozen=True)(dict, x=2)(),
        lambda: FrozenHydrated(),
    ],
)
def test_frozen_via_builds(fn):
    conf_f = fn()

    with pytest.raises(FrozenInstanceError):
        conf_f.x = 3


@given(
    mutable=st.lists(st.integers(), min_size=1)
    | st.dictionaries(st.integers(), st.integers(), min_size=1).map(
        lambda x: {0: 0, **x}
    )
)
def test_mutable_defaults_generated_from_factory(mutable):
    Conf = builds(dict, x=mutable)
    mutable = deepcopy(mutable)

    instance1 = Conf()
    if isinstance(mutable, dict):
        instance1.x.pop(0)
    else:
        instance1.x.pop()

    # mutation via instance1 should not affect other instances of `Conf`
    instance2 = Conf()
    assert instance2.x == mutable

    # make sure hydra behavior is appropriate
    out_Conf = instantiate(Conf)["x"]
    assert out_Conf == mutable

    out_inst = instantiate(instance2)["x"]
    assert out_inst == mutable


class NotSet:
    pass


@pytest.mark.parametrize("recursive", [True, False, NotSet])
@pytest.mark.parametrize("convert", ["none", "partial", "all", NotSet])
@pytest.mark.parametrize("via_builds", [True, False])
def test_hydra_settings_can_be_inherited(recursive, convert, via_builds):
    kwargs = {}
    if recursive is not NotSet:
        kwargs["hydra_recursive"] = recursive

    if convert is not NotSet:
        kwargs["hydra_convert"] = convert

    if via_builds:
        Base = builds(dict, **kwargs)  # type: ignore
        Child = builds(dict, builds_bases=(Base,))  # type: ignore
    else:

        @hydrated_dataclass(target=dict, **kwargs)
        class Base:
            pass

        @hydrated_dataclass(target=dict)
        class Child(Base):
            pass

    if recursive is not NotSet:
        assert Child._recursive_ is Base._recursive_  # type: ignore
    else:
        assert not hasattr(Child, "_recursive_")

    if convert is not NotSet:
        assert Child._convert_ is Base._convert_  # type: ignore
    else:
        assert not hasattr(Child, "_convert_")


@given(
    target=st.sampled_from([int, str]),
    zen_partial=st.none() | st.booleans(),
    name=st.just("CustomName"),
)
def test_dataclass_name(target, zen_partial, name):
    Conf = builds(target, zen_partial=zen_partial, zen_dataclass={"cls_name": name})
    if name is not None:
        assert Conf.__name__ == "CustomName"
        return
    target_name = "str" if target is str else "int"
    if zen_partial is True:
        assert Conf.__name__ == f"PartialBuilds_{target_name}"
    else:
        assert Conf.__name__ == f"Builds_{target_name}"


@dataclass
class VanillaDataClass:
    x: int = 2
    y: str = "a"


@hydrated_dataclass(dict)
class PickleHydrated:
    x: int = 2
    y: str = "a"


PickleBuilds = builds(
    dict,
    x=2,
    y="a",
    zen_dataclass={
        "module": "tests.test_dataclass_semantics",
        "cls_name": "PickleBuilds",
    },
)

PickleCustomBuilds = make_custom_builds_fn(
    zen_dataclass={
        "module": "tests.test_dataclass_semantics",
        "cls_name": "PickleCustomBuilds",
    }
)(dict, x=2, y="a")

PickleMakeConfig = make_config(
    x=2,
    y="a",
    zen_dataclass={
        "module": "tests.test_dataclass_semantics",
        "cls_name": "PickleMakeConfig",
    },
)


PickleJustDataclass = just(
    VanillaDataClass(),
    zen_dataclass={
        "module": "tests.test_dataclass_semantics",
        "cls_name": "PickleJustDataclass",
    },
)


@pytest.mark.parametrize(
    "Conf",
    [
        PickleBuilds,
        PickleCustomBuilds,
        PickleMakeConfig,
        PickleJustDataclass,
        PickleHydrated,
        pytest.param(
            builds(dict, x=2, y="a"),
            marks=pytest.mark.xfail(
                reason="not pickle compatible", raises=PicklingError
            ),
        ),
    ],
)
def test_pickleable(Conf):
    try:
        assert loads(dumps(Conf(y="b"))) != Conf()
    except TypeError:
        pass
    assert loads(dumps(Conf())) == Conf()
    assert loads(dumps(Conf)) is Conf


def test_pickle_just():
    just_int = just(int)
    assert loads(dumps(just_int)) == just_int


def test_hashable_just():
    just_int = just(int)
    assert just_int.__hash__ is not None


def hydrated_fn(zen_dataclass, target=dict):
    @hydrated_dataclass(target, **zen_dataclass)
    class A:
        x: int = 1

    return A


@given(unsafe_hash=...)
@pytest.mark.parametrize(
    "fn",
    [
        lambda **kw: builds(dict, **kw),
        lambda **kw: make_custom_builds_fn(**kw)(dict),
        lambda **kw: just(VanillaDataClass(), **kw),
        lambda **kw: hydrated_fn(**kw),
        make_config,
        hydrated_fn,
    ],
)
def test_hashable(unsafe_hash: Optional[bool], fn):
    kw = {"unsafe_hash": unsafe_hash} if unsafe_hash is not None else {}
    Conf = fn(zen_dataclass=kw)
    assert (Conf.__hash__ is None) is (unsafe_hash is False)


@pytest.mark.parametrize(
    "fn",
    [
        lambda **kw: builds(dict, **kw),
        lambda **kw: make_custom_builds_fn(**kw)(dict),
        lambda **kw: just(VanillaDataClass(), **kw),
        make_config,
    ],
)
def test_namespace(fn):
    conf = fn(zen_dataclass={"namespace": {"fn": lambda _, x: x + 2}})()
    assert conf.fn(2) == 4


@given(kw_only=...)
@pytest.mark.parametrize(
    "fn",
    [
        lambda **kw: builds(dict, x=1, **kw),
        lambda **kw: make_custom_builds_fn(**kw)(dict, x=1),
        lambda **kw: make_config(x=1, **kw),
        lambda **kw: just(VanillaDataClass(), **kw),
        lambda **kw: hydrated_fn(**kw),
    ],
)
def test_kwonly(kw_only: bool, fn):
    Conf = fn(zen_dataclass={"kw_only": kw_only})

    if sys.version_info < (3, 10):
        _ = Conf(2)
    elif kw_only is True:
        with pytest.raises(Exception):
            _ = Conf(2)

    _ = Conf(x=2)


@pytest.mark.parametrize(
    "fn",
    [
        lambda **kw: builds(dict, **kw),
        lambda **kw: make_custom_builds_fn(**kw)(dict),
        lambda **kw: just(VanillaDataClass(), **kw),
        make_config,
    ],
)
def test_bases(fn):
    A = fn()
    B = fn(zen_dataclass={"bases": (A,)})
    C = fn(zen_dataclass={"bases": (B,), "eq": True})
    assert issubclass(B, A)
    assert issubclass(C, A)
    assert issubclass(C, B)


@pytest.mark.parametrize(
    "fn",
    [
        lambda **kw: builds(dict, **kw),
        lambda **kw: make_custom_builds_fn(**kw),
        make_config,
    ],
)
def test_frozen_deprecated(fn):
    with pytest.warns(HydraZenDeprecationWarning):
        fn(frozen=True)


@pytest.mark.parametrize(
    "fn",
    [
        lambda **kw: builds(dict, **kw),
        lambda **kw: make_config(config_name=kw["dataclass_name"]),
    ],
)
def test_dataclassname_deprecated(fn):
    with pytest.warns(HydraZenDeprecationWarning):
        fn(dataclass_name="hi")


def test_unhashable_dataclass_supported():
    from dataclasses import dataclass

    @dataclass(unsafe_hash=False)
    class Unhash:
        x: int

    unhash = Unhash(1)
    Conf = builds(dict, y=unhash, zen_convert={"dataclass": False})
    assert not hasattr(Conf, "y")
    assert Conf().y is unhash


@pytest.mark.parametrize(
    "Conf",
    [
        make_config(zen_dataclass={"module": None}),
        builds(dict, zen_dataclass={"module": None}),
        make_custom_builds_fn(zen_dataclass={"module": None})(dict),
    ],
)
def test_module_is_None(Conf):
    if sys.version_info < (3, 12):
        assert Conf.__module__ == "types"
    else:
        assert Conf.__module__.startswith("hydra_zen")


@pytest.mark.parametrize(
    "Conf",
    [
        make_config(),
        builds(
            dict,
        ),
        make_custom_builds_fn()(dict),
    ],
)
def test_module_is_not_specified(Conf):
    assert Conf.__module__ == "types"


@pytest.mark.parametrize(
    "Conf",
    [
        make_config(zen_dataclass={"module": "aaa"}),
        builds(dict, zen_dataclass={"module": "aaa"}),
        make_custom_builds_fn(zen_dataclass={"module": "aaa"})(dict),
    ],
)
def test_modules_is_specified(Conf):
    assert Conf.__module__ == "aaa"


@pytest.mark.parametrize(
    "cfg",
    [
        builds(dict, x=1),
        builds(dict, x=2)(),
    ],
)
def test_builds_is_copyable(cfg):
    assert to_yaml(cfg) == to_yaml(deepcopy(cfg))
