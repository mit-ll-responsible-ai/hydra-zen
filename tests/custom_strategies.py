# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import string
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import hypothesis.strategies as st

from hydra_zen import ZenStore, builds
from hydra_zen._compatibility import HYDRA_SUPPORTS_OBJECT_CONVERT
from hydra_zen.structured_configs._utils import get_obj_path
from hydra_zen.typing import DataclassOptions
from hydra_zen.typing._implementations import ZenConvert

__all__ = ["valid_builds_args", "partitions", "everything_except"]

_Sequence = Union[List, Tuple, Deque]
T = TypeVar("T", bound=Union[_Sequence, Dict[str, Any]])


def _wrapper(obj):
    return obj


# strategies for drawing valid inputs to `zen_wrappers`
single_wrapper_strat = (
    st.just(_wrapper)
    | st.just(get_obj_path(_wrapper))
    | st.just(_wrapper).map(lambda x: builds(x, zen_partial=True))
)
wrapper_strat = single_wrapper_strat | st.lists(single_wrapper_strat)

slots_strat = st.booleans()


def _compat_slots(conf: Dict[str, Any]):
    # dataclass has some hard rules about a frozen dataclass inheriting
    # from a non-frozen one anf vice versa. Let's avoid this

    if conf.get("weakref_slot", None) is True:
        conf["slots"] = True
    return conf


st.register_type_strategy(
    DataclassOptions,
    st.fixed_dictionaries(
        {},
        optional={
            "cls_name": st.text("abcdefg_", min_size=1),
            "init": st.booleans(),
            "repr": st.booleans(),
            "eq": st.just(True),
            "order": st.booleans(),
            "unsafe_hash": st.booleans(),
            "frozen": st.booleans(),
            "match_args": st.booleans(),
            "kw_only": st.booleans(),
            "slots": st.shared(st.booleans(), key="slot"),
            "weakref_slot": st.shared(st.booleans(), key="slot"),
        },
    ).map(_compat_slots),
)

_valid_builds_strats = dict(
    zen_partial=st.none() | st.booleans(),
    zen_wrappers=wrapper_strat,
    zen_meta=st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1, max_size=2).map(lambda x: "_" + x),
        st.integers(),
    ),
    populate_full_signature=st.booleans(),
    hydra_recursive=st.booleans(),
    hydra_convert=st.sampled_from(
        [
            "none",
            "partial",
            "all",
            *(("object",) if HYDRA_SUPPORTS_OBJECT_CONVERT else ()),
        ]
    ),
    builds_bases=st.just(()),
    zen_convert=st.none() | st.from_type(ZenConvert),
    zen_dataclass=st.none() | st.from_type(DataclassOptions),
)


def _compat_frozen(conf: Dict[str, Any]):
    # dataclass has some hard rules about a frozen dataclass inheriting
    # from a non-frozen one anf vice versa. Let's avoid this
    if conf.get("frozen", None) is True and conf.get("builds_bases", ()):
        conf["frozen"] = False
    return conf


def valid_builds_args(*required: str, excluded: Sequence[str] = ()):
    """Generates valid inputs for all nameable args in `builds`, except `dataclass_name`."""
    assert len(required) == len(set(required))
    _required = set(required)
    _excluded = set(excluded)
    assert _required.isdisjoint(_excluded)

    assert _required <= set(_valid_builds_strats), _required - set(_valid_builds_strats)
    assert _excluded <= set(_valid_builds_strats), _excluded - set(_valid_builds_strats)

    return (
        st.fixed_dictionaries(
            {k: _valid_builds_strats[k] for k in sorted(_required)},
            optional={
                k: v
                for k, v in _valid_builds_strats.items()
                if k not in _excluded and k not in _required
            },
        )
        .map(_compat_frozen)
        .map(_compat_slots)
    )


@st.composite
def _partition(draw: st.DrawFn, collection: T, ordered: bool) -> Tuple[T, T]:

    if isinstance(collection, dict):
        keys = list(collection)
    else:
        keys = list(range(len(collection)))

    divider = draw(st.integers(0, len(keys)))

    if not ordered:
        keys = draw(st.permutations(keys))

    keys_a, keys_b = keys[divider:], keys[:divider]
    if not isinstance(collection, dict):
        caster = type(collection)
        return tuple((caster(collection[k] for k in keys)) for keys in [keys_a, keys_b])  # type: ignore
    else:
        return tuple(({k: collection[k] for k in keys}) for keys in [keys_a, keys_b])  # type: ignore


def partitions(
    collection: Union[T, st.SearchStrategy[T]], ordered: bool = True
) -> st.SearchStrategy[Tuple[T, T]]:
    """Randomly partitions a collection or dictionary into two partitions."""
    if isinstance(collection, st.SearchStrategy):
        return collection.flatmap(lambda x: _partition(x, ordered=ordered))
    return cast(st.SearchStrategy[Tuple[T, T]], _partition(collection, ordered))


def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def f():
    pass


f_Build = builds(f, zen_dataclass={"cls_name": "f_Build"})


@st.composite
def store_entries(draw: st.DrawFn):
    group = draw(st.lists(st.sampled_from("abcde")).map(lambda x: None if not x else x))
    if group is not None:
        group = "/".join(group)
    name = draw(st.uuids().map(str))
    package = draw(st.none() | st.sampled_from("xyz"))
    target = draw(st.sampled_from([{"a": 1}, f_Build, f]))
    return (target, {"name": name, "group": group, "package": package})


@st.composite
def new_stores(draw: st.DrawFn, deferred_hydra_store: Optional[bool] = None):
    name = draw(st.none() | st.sampled_from(["foo", "bar", "baz"]))
    return ZenStore(
        name,
        deferred_hydra_store=draw(
            st.booleans().map(lambda x: not x), label="deferred_hydra_store"
        )
        if deferred_hydra_store is None
        else deferred_hydra_store,
        overwrite_ok=draw(st.booleans(), label="overwrite_ok"),
        deferred_to_config=draw(
            st.booleans().map(lambda x: not x), label="deferred_to_config"
        ),
    )


@st.composite
def stores(draw: st.DrawFn):
    """Returned stores always defer storing to hydra store to prevent unexpected
    change in global state"""
    store = draw(new_stores(deferred_hydra_store=True))
    for target, kw in draw(st.lists(store_entries())):
        store(target, **kw)
    return store


st.register_type_strategy(ZenStore, stores())
