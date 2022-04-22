import inspect

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds
from tests.custom_strategies import _valid_builds_strats, partitions, valid_builds_args


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
    assert nameable_builds_args - {"dataclass_name", "hydra_defaults"} == set(
        _valid_builds_strats
    )


@given(req_excl=partitions(tuple(_valid_builds_strats), ordered=False), data=st.data())
def test_valid_builds_excluded_and_required(req_excl, data: st.DataObject):
    required, excluded = req_excl
    drawn_args = data.draw(
        valid_builds_args(*required, excluded=excluded).map(set), label="draw_args"
    )
    assert set(required) <= drawn_args
    assert drawn_args.isdisjoint(set(excluded))


a_list = [1, 2, 3]
a_dict = dict(a=-1, b=-2, c=-3)


@given(
    collection_or_strat=st.sampled_from([a_list, a_dict])
    | st.sampled_from([a_list, a_dict]).map(st.just),
    ordered=st.booleans(),
    data=st.data(),
)
def test_partition(collection_or_strat, ordered: bool, data: st.DataObject):
    a, b = data.draw(partitions(collection_or_strat, ordered=ordered))
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
