# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Hashable, Optional

import pytest
from hydra.core.config_store import ConfigStore
from hypothesis import given
from omegaconf import DictConfig, ListConfig

from hydra_zen import builds, instantiate, just, make_config, store
from hydra_zen._compatibility import HYDRA_SUPPORTS_LIST_INSTANTIATION
from hydra_zen.wrapper._implementations import ZenStore

cs = ConfigStore().instance()


def func(a: int, b: int):
    assert isinstance(a, int)
    assert isinstance(b, int)
    return (a, b)


def instantiate_from_repo(
    name: str,
    group: Optional[str] = None,
    package: Optional[str] = None,
    provider: Optional[str] = None,
    **kw,
):
    """Fetches config-node from repo by name and instantiates it
    with provided kwargs"""
    if group is not None:
        repo = cs.repo
        for group_name in group.split("/"):
            repo = repo[group_name]
        item = repo[f"{name}.yaml"]
    else:
        item = cs.repo[f"{name}.yaml"]
    assert item.package == package
    assert item.provider == provider
    return instantiate(item.node, **kw)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, a=1, b=2), id="inline"),
        pytest.param(lambda: store(a=1, b=2)(func), id="decorated"),
        pytest.param(lambda: store()(func, a=1, b=2), id="partiald_inline"),
        pytest.param(lambda: store()(a=1, b=2)(func), id="partiald_decorated"),
        pytest.param(lambda: store(a=-22)(a=1, b=-22)(func, b=2), id="kw_overrides"),
        pytest.param(
            lambda: store(name="BAD")(func),
            id="ensure_can_fail1",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(a=22, b=10)(func),
            id="ensure_can_fail2",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_kw_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    store.add_to_hydra_store()
    assert out is func
    assert instantiate_from_repo("func") == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, name="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(name="dunk")(func), id="decorated"),
        pytest.param(lambda: store(name="O1")(func, name="dunk"), id="partiald_inline"),
        pytest.param(
            lambda: store(name="O1")(name="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(name="O1")(name="O2")(func, name="dunk"), id="kw_overrides"
        ),
        pytest.param(
            lambda: store(name="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_name_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    store.add_to_hydra_store()
    assert instantiate_from_repo("dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, group="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(group="dunk")(func), id="decorated"),
        pytest.param(
            lambda: store(group="O1")(func, group="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: store(group="O1")(group="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(group="O1")(group="O2")(func, group="dunk"), id="kw_overrides"
        ),
        pytest.param(
            lambda: store(group="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_group_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    store.add_to_hydra_store()
    assert instantiate_from_repo(name="func", group="dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, package="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(package="dunk")(func), id="decorated"),
        pytest.param(
            lambda: store(package="O1")(func, package="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: store(package="O1")(package="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(package="O1")(package="O2")(func, package="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: store(package="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_package_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    store.add_to_hydra_store()
    assert instantiate_from_repo(name="func", package="dunk", a=1, b=2) == (1, 2)


def special_fn(x, **kw):
    kw["target"] = just(x)
    return kw


def never_call(*a, **k):
    assert False


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, to_config=special_fn, a=1, b=2), id="inline"),
        pytest.param(
            lambda: store(to_config=special_fn, a=1, b=2)(func),
            id="decorated",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=1, b=2)(func, to_config=special_fn),
            id="partiald_inline",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=1, b=2)(to_config=special_fn)(func),
            id="partiald_decorated",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=22, b=2)(to_config=never_call)(
                func, to_config=special_fn, a=1
            ),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=1, b=2)(func),
            id="ensure_can_fail1",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(func, to_config=never_call, a=1, b=2),
            id="ensure_can_fail2",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(func, a=1, b=2),
            id="ensure_can_fail3",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(func, a=1, b=3, to_config=special_fn),
            id="ensure_can_fail4",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_to_config_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    store.add_to_hydra_store()
    assert instantiate_from_repo(name="func") == dict(a=1, b=2, target=func)


def override_store(
    name: str,
    node: Any,
    group: Optional[str] = None,
    package: Optional[str] = None,
    provider: Optional[str] = None,
):
    assert False


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, provider="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(provider="dunk")(func), id="decorated"),
        pytest.param(
            lambda: store(provider="O1")(func, provider="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: store(provider="O1")(provider="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(provider="O1")(provider="O2")(func, provider="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: store(provider="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_provider_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    store.add_to_hydra_store()
    assert instantiate_from_repo(name="func", provider="dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize("include_none", [True, False])
@pytest.mark.usefixtures("clean_store")
def test_store_nested_groups(include_none: bool):
    # Tests that nested groups are stored in Hydra store as-expected
    # and that ZenStore's __getitem__ has parity with the store
    local_store = ZenStore(deferred_hydra_store=False)
    if include_none:
        local_store({"a": 0}, name="a")
    local_store({"a": 1}, group="A", name="a")
    local_store({"a": 2}, group="A", name="b")
    local_store({"a": 3}, group="A/B", name="ab")
    local_store({"a": 4}, group="A/B/C", name="abc")

    if include_none:
        assert instantiate_from_repo(name="a") == instantiate(local_store[None, "a"])
        assert instantiate_from_repo(name="a") == {"a": 0}

    assert instantiate_from_repo(name="a", group="A") == instantiate(
        local_store["A", "a"]
    )
    assert instantiate_from_repo(name="a", group="A") == {"a": 1}

    assert instantiate_from_repo(name="b", group="A") == instantiate(
        local_store["A", "b"]
    )

    assert instantiate_from_repo(name="b", group="A") == {"a": 2}

    assert instantiate_from_repo(name="ab", group="A/B") == instantiate(
        local_store["A/B", "ab"]
    )
    assert instantiate_from_repo(name="ab", group="A/B") == {"a": 3}

    assert instantiate_from_repo(name="abc", group="A/B/C") == instantiate(
        local_store["A/B/C", "abc"]
    )
    assert instantiate_from_repo(name="abc", group="A/B/C") == {"a": 4}

    if include_none:
        assert local_store.groups == [None, "A", "A/B", "A/B/C"]
    else:
        assert local_store.groups == ["A", "A/B", "A/B/C"]
    assert len(local_store[None]) == 1
    assert len(local_store["A"]) == 4
    assert len(local_store["A/B"]) == 2
    assert len(local_store["A/B/C"]) == 1

    assert set(local_store["A/B"]) < set(local_store["A"])
    assert set(local_store["A/B/C"]) < set(local_store["A/B"])


@pytest.mark.parametrize("bad_val", [1, True, ("a",)])
@pytest.mark.parametrize("field_name", ["name", "group", "package"])
@pytest.mark.usefixtures("clean_store")
def test_store_param_validation(bad_val, field_name: str):
    with pytest.raises(TypeError, match=rf"`{field_name}` must be"):
        store(func, **{field_name: bad_val})
        store.add_to_hydra_store()


@dataclass
class DC:
    ...


dc = DC()


def test_validate_get_name():
    with pytest.raises(TypeError, match=r"Cannot infer config store entry name"):
        store(dc)


@pytest.mark.parametrize("name1", "ab")
@pytest.mark.parametrize("name2", "ab")
@pytest.mark.parametrize("group1", [None, "c", "d", "e/f", "e/f/g"])
@pytest.mark.parametrize("group2", [None, "c", "d", "e/f", "e/f/g"])
@pytest.mark.usefixtures("clean_store")
def test_raise_on_redundant_store(
    name1: str, name2: str, group1: Optional[str], group2: Optional[str]
):
    _store = ZenStore(overwrite_ok=False)

    _store({"a": 1}, name=name1, group=group1)
    if (name1, group1) == (name2, group2):
        with pytest.raises(ValueError):
            _store({"b": 2}, name=name2, group=group2)
    else:
        _store({"b": 2}, name=name2, group=group2)


@pytest.mark.parametrize("name", "ab")
@pytest.mark.parametrize("group", [None, "c", "d", "e/f", "e/f/g"])
@pytest.mark.parametrize("outer", [True, False])
@pytest.mark.parametrize("inner", [True, False])
@pytest.mark.usefixtures("clean_store")
def test_overwrite_ok(outer: bool, inner: bool, name, group):
    _store = ZenStore(overwrite_ok=outer)
    _store({}, name=name, group=group)
    if not outer:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"(name={name} group={group}): Hydra config store entry already exists"
            ),
        ):
            _store({}, name=name, group=group)
        return
    _store({}, name=name, group=group)
    if not inner:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"(name={name} group={group}): Hydra config store entry already exists"
            ),
        ):
            _store.add_to_hydra_store(overwrite_ok=inner)
    else:
        _store.add_to_hydra_store(overwrite_ok=inner)


@pytest.mark.parametrize(
    "target",
    [
        {"a": 1},
        ["a", "b"],
        DictConfig({"a": 1}),
        ListConfig(["a", "b"]),
        make_config(),
        make_config()(),
        builds(dict, a=1),
        DC,
        dc,
        partial(func, a=1, b=1),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_default_to_config_produces_instantiable_configs(target):
    if not HYDRA_SUPPORTS_LIST_INSTANTIATION and isinstance(target, (list, ListConfig)):
        pytest.xfail("Hydra doesn't support list instantiation")
    store(target, name="target")
    store.add_to_hydra_store()
    instantiate_from_repo("target")


class NoHydra(ZenStore):
    # needed for hypothesis test -- we can't use clean_store fixture
    # per-test
    def add_to_hydra_store(self, overwrite_ok: Optional[bool] = None):
        return


@given(...)
def test_self_partialing_reflects_mutable_state(zstore: NoHydra):
    zstore2 = zstore()
    zstore3 = zstore(a=22)
    stores = [zstore2, zstore3]

    for attr in zstore.__slots__:
        if attr == "_defaults":
            continue

        for s in stores:
            assert getattr(zstore, attr) == getattr(s, attr)
    # assert not zstore._internal_repo
    zstore(dict(a=1), name="a")
    zstore2(dict(a=2), name="b")
    zstore3(dict(a=3), name="c")

    assert len(zstore._queue) == 3
    assert len(zstore._internal_repo) == 3
    for attr in zstore.__slots__:
        x = getattr(zstore, attr)
        if not isinstance(x, Hashable):
            continue
        assert x is getattr(zstore2, attr)
        assert x is getattr(zstore3, attr)


@given(...)
def test_stores_have_independent_mutable_state(store1: ZenStore, store2: ZenStore):
    for attr in store1.__slots__:
        x = getattr(store1, attr)
        if isinstance(x, Hashable):
            continue
        assert x is not getattr(store2, attr)
        assert x is not getattr(store, attr)  # check default store


@pytest.mark.parametrize("name", "ab")
@pytest.mark.parametrize("group", [None, "c", "d", "e/f", "e/f/g"])
@pytest.mark.usefixtures("clean_store")
def test_deferred_to_config(name, group):
    Store = partial(ZenStore, deferred_hydra_store=True)

    s = Store(deferred_to_config=True)
    s(1, name=name, group=group, to_config=never_call)
    with pytest.raises(AssertionError):
        s.add_to_hydra_store()

    s = Store(deferred_to_config=True)
    s(1, name=name, group=group, to_config=never_call)
    with pytest.raises(AssertionError):
        s[group, name]

    s = ZenStore(deferred_to_config=False)
    s = s(to_config=never_call)
    with pytest.raises(AssertionError):
        s(dict(a=1), name=name, group=group)


def test_self_partialing_preserves_subclass():
    s1 = NoHydra()
    s2 = s1()
    assert s1 is not s2
    assert isinstance(s2, NoHydra)


@pytest.mark.parametrize("name", "ab")
@pytest.mark.parametrize("deferred_to_config", [True, False])
@pytest.mark.usefixtures("clean_store")
def test_getitem(deferred_to_config: bool, name: str):
    s = ZenStore(deferred_to_config=deferred_to_config)
    conf = make_config()()
    s(conf, name=name)
    assert s[None, name] is conf


@pytest.mark.usefixtures("clean_store")
def test_default_to_config_validates_dataclass_instance_with_kw():
    store(make_config(a=2)(), name="dc", a=1)

    with pytest.raises(ValueError):
        store[None, "dc"]


@pytest.mark.parametrize(
    "param_name", ["deferred_to_config", "deferred_hydra_store", "overwrite_ok"]
)
def test_validate_init(param_name):
    with pytest.raises(TypeError, match=f"{param_name} must be a bool"):
        ZenStore(**{param_name: "bad"})
