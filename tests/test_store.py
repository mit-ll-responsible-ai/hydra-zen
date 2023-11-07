# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import os
import re
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Hashable, Optional

import hypothesis.strategies as st
import pytest
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hypothesis import assume, given, note, settings
from omegaconf import DictConfig, ListConfig

from hydra_zen import (
    ZenStore,
    builds,
    hydrated_dataclass,
    instantiate,
    just,
    make_config,
    store as default_store,
)
from tests.custom_strategies import new_stores, store_entries

cs = ConfigStore().instance()


@contextmanager
def clean_store():
    """Provides access to configstore repo and restores state after test"""
    prev_state = deepcopy(cs.repo)
    zen_prev_state = (default_store._internal_repo.copy(), default_store._queue.copy())
    try:
        yield cs.repo
    finally:
        cs.repo = prev_state
        int_repo, queue = zen_prev_state
        default_store._internal_repo = int_repo
        default_store._queue = queue


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
        pytest.param(lambda: default_store(func, a=1, b=2), id="inline"),
        pytest.param(lambda: default_store(a=1, b=2)(func), id="decorated"),
        pytest.param(lambda: default_store()(func, a=1, b=2), id="partiald_inline"),
        pytest.param(lambda: default_store()(a=1, b=2)(func), id="partiald_decorated"),
        pytest.param(
            lambda: default_store(a=-22)(a=1, b=-22)(func, b=2), id="kw_overrides"
        ),
        pytest.param(
            lambda: default_store(name="BAD")(func),
            id="ensure_can_fail1",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: default_store(a=22, b=10)(func),
            id="ensure_can_fail2",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_kw_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    default_store.add_to_hydra_store()
    assert out is func
    assert instantiate_from_repo("func") == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: default_store(func, name="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: default_store(name="dunk")(func), id="decorated"),
        pytest.param(
            lambda: default_store(name="O1")(func, name="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: default_store(name="O1")(name="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: default_store(name="O1")(name="O2")(func, name="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: default_store(name="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_name_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    default_store.add_to_hydra_store()
    assert instantiate_from_repo("dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: default_store(func, group="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: default_store(group="dunk")(func), id="decorated"),
        pytest.param(
            lambda: default_store(group="O1")(func, group="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: default_store(group="O1")(group="dunk")(func),
            id="partiald_decorated",
        ),
        pytest.param(
            lambda: default_store(group="O1")(group="O2")(func, group="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: default_store(group="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_group_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    default_store.add_to_hydra_store()
    assert instantiate_from_repo(name="func", group="dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(
            lambda: default_store(func, package="dunk", a=1, b=2), id="inline"
        ),
        pytest.param(lambda: default_store(package="dunk")(func), id="decorated"),
        pytest.param(
            lambda: default_store(package="O1")(func, package="dunk"),
            id="partiald_inline",
        ),
        pytest.param(
            lambda: default_store(package="O1")(package="dunk")(func),
            id="partiald_decorated",
        ),
        pytest.param(
            lambda: default_store(package="O1")(package="O2")(func, package="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: default_store(package="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_package_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    default_store.add_to_hydra_store()
    assert instantiate_from_repo(name="func", package="dunk", a=1, b=2) == (1, 2)


def special_fn(x, **kw):
    kw["target"] = just(x)
    return kw


def never_call(*a, **k):
    assert False


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(
            lambda: default_store(func, to_config=special_fn, a=1, b=2), id="inline"
        ),
        pytest.param(
            lambda: default_store(to_config=special_fn, a=1, b=2)(func),
            id="decorated",
        ),
        pytest.param(
            lambda: default_store(to_config=never_call, a=1, b=2)(
                func, to_config=special_fn
            ),
            id="partiald_inline",
        ),
        pytest.param(
            lambda: default_store(to_config=never_call, a=1, b=2)(to_config=special_fn)(
                func
            ),
            id="partiald_decorated",
        ),
        pytest.param(
            lambda: default_store(to_config=never_call, a=22, b=2)(
                to_config=never_call
            )(func, to_config=special_fn, a=1),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: default_store(to_config=never_call, a=1, b=2)(func),
            id="ensure_can_fail1",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: default_store(func, to_config=never_call, a=1, b=2),
            id="ensure_can_fail2",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: default_store(func, a=1, b=2),
            id="ensure_can_fail3",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: default_store(func, a=1, b=3, to_config=special_fn),
            id="ensure_can_fail4",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_to_config_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    default_store.add_to_hydra_store()
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
        pytest.param(
            lambda: default_store(func, provider="dunk", a=1, b=2), id="inline"
        ),
        pytest.param(lambda: default_store(provider="dunk")(func), id="decorated"),
        pytest.param(
            lambda: default_store(provider="O1")(func, provider="dunk"),
            id="partiald_inline",
        ),
        pytest.param(
            lambda: default_store(provider="O1")(provider="dunk")(func),
            id="partiald_decorated",
        ),
        pytest.param(
            lambda: default_store(provider="O1")(provider="O2")(func, provider="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: default_store(provider="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
def test_provider_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    default_store.add_to_hydra_store()
    assert instantiate_from_repo(name="func", provider="dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize("include_none", [True, False])
@pytest.mark.usefixtures("clean_store")
def test_store_nested_groups(include_none: bool):
    # Tests that nested groups are stored in Hydra store as-expected
    # and that ZenStore's __getitem__ has parity with the store
    local_store = ZenStore(deferred_hydra_store=False)
    if include_none:
        local_store({"a": 0}, name="a")

    def check_repr():
        assert isinstance(repr(local_store), str)

    check_repr()

    local_store({"a": 1}, group="A", name="a")
    check_repr()

    local_store({"a": 2}, group="A", name="b")
    check_repr()

    local_store({"a": 3}, group="A/B", name="ab")
    check_repr()

    local_store({"a": 4}, group="A/B/C", name="abc")
    check_repr()

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
        assert len(local_store[None]) == 1
    else:
        assert local_store.groups == ["A", "A/B", "A/B/C"]
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
        default_store(func, **{field_name: bad_val})
        default_store.add_to_hydra_store()


@dataclass
class DC:
    ...


dc = DC()


def test_validate_get_name():
    with pytest.raises(TypeError, match=r"Cannot infer config store entry name"):
        default_store(dc)


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
            match=re.escape(f"(name={name} group={group}): Store entry already exists"),
        ):
            _store({}, name=name, group=group)
        return
    _store.add_to_hydra_store(overwrite_ok=True)
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
    default_store(target, name="target")
    default_store.add_to_hydra_store()
    instantiate_from_repo("target")


@given(zstore=new_stores())
def test_self_partialing_reflects_mutable_state(zstore: ZenStore):
    with clean_store():
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

        assert zstore == zstore2
        assert zstore == zstore3
        assert zstore2 == zstore3

        assert len(zstore._queue) == (3 if zstore._deferred_store else 0)
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
        assert x is not getattr(default_store, attr)  # check default store


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
    class SubStore(ZenStore):
        ...

    s1 = SubStore()
    s2 = s1()
    assert s1 is not s2
    assert isinstance(s2, SubStore)


@pytest.mark.usefixtures("clean_store")
def test_default_to_config_validates_dataclass_instance_with_kw():
    default_store(make_config(a=2)(), name="dc", a=1)

    with pytest.raises(ValueError):
        default_store[None, "dc"]


@pytest.mark.parametrize(
    "param_name", ["deferred_to_config", "deferred_hydra_store", "overwrite_ok"]
)
def test_validate_init(param_name):
    with pytest.raises(TypeError, match=f"{param_name} must be a bool"):
        ZenStore(**{param_name: "bad"})  # type: ignore


def contains(key, store_):
    assert key in store_ and store_[key]


def not_contains(key, store_):
    assert key not in store_
    try:
        v = store_["key"]
    except KeyError:
        return
    assert not v


@pytest.mark.usefixtures("clean_store")
def test_contains_manual():
    _store = ZenStore()
    _store({"": 2}, name="grape")
    _store({"": 1}, group="a/b", name="apple")
    assert_contains = partial(contains, store_=_store)
    assert_not_contains = partial(not_contains, store_=_store)
    assert_contains("a")
    assert_contains("a/b")
    assert_not_contains("b")
    assert_not_contains("b/a")
    assert_not_contains("c")
    assert_not_contains("a/c")
    assert_not_contains("a/b/c")
    assert_not_contains("a/c/b")
    assert_contains(("a/b", "apple"))
    assert_not_contains(("a", "apple"))
    assert_not_contains(("a/b", "pear"))
    assert_contains(None)
    assert_contains((None, "grape"))
    assert_not_contains((None, "apple"))
    assert_not_contains(1)
    assert_not_contains((1, 2))
    assert_not_contains(("a", "apple", "grape"))


@given(...)
def test_contains_consistent_with_getitem(store: ZenStore):
    assert "NOTAGROUP" not in store
    assert "NOTAGROUP/" not in store
    assert "/NOTAGROUP" not in store
    for entry in store:
        group = entry["group"]
        name = entry["name"]
        assert group in store
        assert (group, name) in store
        assert name not in store
        assert not store[name]

        assert (name, group) not in store  # type: ignore
        with pytest.raises(KeyError):
            store[name, group]  # type: ignore

        if group is None:
            continue

        group_parts = group.split("/")
        for n in range(1, len(group_parts) + 1):
            assert "/".join(group_parts[:n]) in store


@settings(deadline=None)
@given(entries=st.lists(store_entries()), sub_store=st.booleans())
def test_iter(entries, sub_store):
    with clean_store():
        _store = ZenStore()
        _store_it = _store() if sub_store else _store

        for target, kw in entries:
            _store_it(target, **kw)

        del _store_it

        iter_out = list(_store)
        assert len(iter_out) == len(entries)


@settings(max_examples=10, deadline=None)
@given(entries=st.lists(store_entries()), store=new_stores(), sub_store=st.booleans())
def test_bool(entries, store: ZenStore, sub_store: bool):
    with clean_store():
        _store_it = store() if sub_store else store

        for target, kw in entries:
            _store_it(target, **kw)
        del _store_it

        assert bool(entries) is bool(store)


@settings(max_examples=20, deadline=None)
@given(...)
def test_repr(store: ZenStore):
    assert isinstance(repr(store), str)


@settings(deadline=None)
@given(store=..., num_adds=st.integers(1, 5))
def test_repeated_add_to_hydra_store_ok(store: ZenStore, num_adds: int):
    # store should clear internal queue so that multiple add-to-store calls
    # don't conflict
    assert bool(store) is store.has_enqueued()
    with clean_store():
        for _ in range(num_adds):
            store.add_to_hydra_store()
            assert not store.has_enqueued()


@settings(deadline=None)
@given(...)
def test_store_protects_overwriting_entries_in_hydra_store(store: ZenStore):
    with clean_store():
        assume(store and store.has_enqueued())
        entry, *_ = store
        note(f"entry: {entry}")
        note(f"{store._internal_repo}, {store._queue}")
        cs.store(**entry)

        with pytest.raises(ValueError):
            store.add_to_hydra_store(overwrite_ok=False)
        store.add_to_hydra_store(overwrite_ok=True)


@settings(max_examples=20, deadline=None)
@given(...)
def test_getitem(store: ZenStore):
    assume(store)
    with clean_store():
        for entry in store:
            assert store[entry["group"], entry["name"]] is entry["node"]
            assert len(store[entry["group"]]) > 0


@settings(max_examples=20, deadline=None)
@given(...)
def test_eq(store1: ZenStore, store2: ZenStore):
    with clean_store():
        assert store1 != store2
        assert store1 != 1
        assert store1 == store1
        assert store1 == store1(param="blah")

        store1.add_to_hydra_store()
        store2.add_to_hydra_store()
        assert store1 != store2
        assert store1 == store1(param="blah")


@settings(max_examples=20, deadline=None)
@given(...)
def test_get_entry(store: ZenStore):
    assume(store)
    entry, *_ = store
    entry_ = store.get_entry(entry["group"], name=entry["name"])
    assert entry == entry_


@settings(max_examples=20, deadline=None)
@given(store=...)
@pytest.mark.parametrize(
    "getter",
    [
        lambda store, *_: next(iter(store)),
        lambda store, group, name: store.get_entry(group, name),
    ],
)
def test_entry_access_cannot_mutate_store(store: ZenStore, getter):
    assume(store)
    entries = tuple(d.copy() for d in store)

    entry = getter(store, entries[0]["group"], entries[0]["name"])
    entry["name"] = 2222
    new_entries = tuple(d.copy() for d in store)
    assert all(e in new_entries for e in entries)


class CustomHydraConf(HydraConf):
    ...


@pytest.mark.parametrize("conf", [CustomHydraConf, HydraConf()])
@pytest.mark.parametrize("deferred", [True, False])
@pytest.mark.usefixtures("clean_store")
def test_auto_support_for_HydraConf(conf: HydraConf, deferred: bool):
    with clean_store():
        st1 = ZenStore(deferred_hydra_store=deferred)
        st2 = ZenStore(deferred_hydra_store=True)
        st1(conf)
        st1.add_to_hydra_store()
        assert st1["hydra", "config"] is conf

        with pytest.raises(
            ValueError,
            match=re.escape(r"(name=config group=hydra):"),
        ):
            # Attempting to overwrite HydraConf within the same
            # store should fail.
            st1(conf)

        st2(conf)
        with pytest.raises(ValueError):
            st2.add_to_hydra_store()


@pytest.mark.skipif(
    sys.platform.startswith("win") and bool(os.environ.get("CI")),
    reason="Things are weird on GitHub Actions and Windows",
)
@pytest.mark.parametrize(
    "inp",
    [
        None,
        pytest.param(
            "hydra.job.chdir=False",
            marks=pytest.mark.xfail(
                reason="Hydra should not change directory in this case"
            ),
        ),
    ],
)
@pytest.mark.usefixtures("clean_store")
@pytest.mark.usefixtures("cleandir")
def test_configure_hydra_chdir(inp: str):
    import subprocess
    from pathlib import Path

    path = (Path(__file__).parent / "example_app" / "change_hydra_config.py").absolute()

    cli = ["python", path]
    if inp:
        cli.append(inp)
    subprocess.run(cli).check_returncode()


def test_node_warns():
    store = ZenStore(warn_node_kwarg=True)
    with pytest.warns(UserWarning):
        store(node=builds(int))


def foo(x: int):
    return x


def test_store_hydrated_dataclass():
    # regression test for: https://github.com/mit-ll-responsible-ai/hydra-zen/issues/453

    @hydrated_dataclass(foo)
    class SomethingHydrated:
        x: int = 1

    store = ZenStore()
    store(SomethingHydrated, name="foo", x=2)
    assert instantiate(store[None, "foo"]) == 2


def test_del():
    s = ZenStore()
    s({}, name="a")
    s({}, name="b")
    assert len(s) == 2
    assert len(s._queue) == 2

    del s[None, "a"]
    assert len(s) == 1
    assert len(s._queue) == 1
    assert (None, "a") not in s
    assert (None, "a") not in s._queue

    s.delete_entry(None, "b")
    assert not s
    assert not s._queue


def test_copy():
    s = ZenStore(name="s")(group="G")
    s({}, name="a")
    s({}, name="b")

    s2 = s.copy()
    s2({}, name="c")

    assert s != s2

    del s["G", "a"]
    assert len(s) == 1
    assert len(s._queue) == 1
    assert len(s2) == 3
    assert len(s2._queue) == 3
    assert ("G", "c") in s2

    assert s2.name == "s_copy"
    assert s2.copy("moo").name == "moo"
