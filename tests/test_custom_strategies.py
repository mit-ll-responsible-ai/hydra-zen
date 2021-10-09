import inspect

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds
from tests.custom_strategies import _valid_builds_strats, partition, valid_builds_args


def f():
    pass


@pytest.mark.filterwarnings(
    "ignore:A structured config was supplied for `zen_wrappers`"
)
@given(valid_builds_args())
def test_valid_builds_args_passes_builds(kwargs):
    builds(f, **kwargs)


def test_valid_build_strats_are_exhaustive():
    nameable_builds_args = set(
        n
        for n, p in inspect.signature(builds).parameters.items()
        if p.kind is p.KEYWORD_ONLY
    )
    assert nameable_builds_args - {"dataclass_name"} == set(_valid_builds_strats)


a_list = [1, 2, 3]
a_dict = dict(a=-1, b=-2, c=-3)


@given(
    collection_or_strat=st.sampled_from([a_list, a_dict])
    | st.sampled_from([a_list, a_dict]).map(st.just),
    ordered=st.booleans(),
    data=st.data(),
)
def test_partition(collection_or_strat, ordered: bool, data: st.DataObject):
    a, b = data.draw(partition(collection_or_strat, ordered=ordered))
    assert len(set(a)) == len(a)
    assert len(set(b)) == len(b)
    assert len(a) + len(b) == 3
    if ordered:
        assert sorted(a) == list(a)
        assert sorted(b) == list(b)

    if isinstance(a, dict):
        merged = {}
        merged.update(a)
        merged.update(b)
        assert merged == a_dict
    else:
        assert sorted(a + b) == a_list
