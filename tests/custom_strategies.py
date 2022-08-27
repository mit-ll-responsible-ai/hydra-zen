import string
from typing import Any, Deque, Dict, List, Sequence, Tuple, TypeVar, Union, cast

import hypothesis.strategies as st

from hydra_zen import builds
from hydra_zen.structured_configs._utils import get_obj_path
from hydra_zen.typing._implementations import ZenConvert

__all__ = ["valid_builds_args", "partitions"]

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


_valid_builds_strats = dict(
    zen_partial=st.none() | st.booleans(),
    zen_wrappers=wrapper_strat,
    zen_meta=st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1, max_size=2).map(lambda x: "_" + x),
        st.integers(),
    ),
    populate_full_signature=st.booleans(),
    hydra_recursive=st.booleans(),
    hydra_convert=st.sampled_from(["none", "partial", "all"]),
    frozen=st.booleans(),
    builds_bases=st.lists(
        st.sampled_from([builds(x) for x in (int, len, dict)]), unique=True
    ).map(tuple),
    zen_convert=st.none() | st.from_type(ZenConvert),
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

    return st.fixed_dictionaries(
        {k: _valid_builds_strats[k] for k in sorted(_required)},
        optional={
            k: v
            for k, v in _valid_builds_strats.items()
            if k not in _excluded and k not in _required
        },
    ).map(_compat_frozen)


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
        return collection.flatmap(lambda x: _partition(x, ordered=ordered))  # type: ignore
    return cast(st.SearchStrategy[Tuple[T, T]], _partition(collection, ordered))
