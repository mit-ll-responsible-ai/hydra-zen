import string

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from hydra_zen import instantiate, launch, make_config
from hydra_zen._launch import hydra_list, multirun, value_check

any_types = st.from_type(type)


def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


@settings(max_examples=10)
@given(
    name=st.sampled_from(["name_a", "name_b"]),
    target_type=st.shared(any_types, key="target_type"),
    value=st.shared(any_types, key="target_type").flatmap(everything_except),
)
def test_type_catches_bad_type(name, target_type, value):
    with pytest.raises(TypeError, match=rf"`{name}` must be of type\(s\) .*"):
        value_check(name, value=value, type_=target_type)


@given(
    target_type=st.shared(any_types, key="target_type"),
    value=st.shared(any_types, key="target_type").flatmap(st.from_type),
)
def test_type_passes_valid_type(target_type, value):
    value_check("dummy", value=value, type_=target_type)


@pytest.mark.usefixtures("cleandir")
@settings(max_examples=10, deadline=None)
@given(
    int_=st.integers(),
    bool_=st.booleans(),
    float_=st.floats(-10, 10),
    list_=st.lists(st.integers()),
    str_=st.text(alphabet=string.ascii_lowercase).filter(
        lambda x: x != "true" and x != "false"
    ),
    mrun=st.lists(
        st.booleans() | st.lists(st.integers()),
        min_size=2,
        max_size=5,
    ),
)
def test_overrides_roundtrip(
    int_,
    bool_,
    float_,
    str_,
    list_,
    mrun,
):

    overrides = {
        "+int_": int_,
        "+float_": float_,
        "+str_": str_,
        "+bool_": bool_,
        "+list_": hydra_list(list_),
        "+mrun": multirun(mrun),
    }
    (jobs,) = launch(make_config(), instantiate, overrides, multirun=True)

    assert len(jobs) == len(mrun)
    for i, job in enumerate(jobs):
        assert job.return_value.int_ == int_
        assert job.return_value.bool_ == bool_
        assert job.return_value.str_ == str_
        assert job.return_value.list_ == list_
        assert job.return_value.mrun == mrun[i]
