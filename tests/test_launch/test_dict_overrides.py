import string
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from omegaconf import ListConfig

from hydra_zen import instantiate, launch, load_from_yaml, make_config
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
    none_=st.none(),
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
    none_,
    int_,
    bool_,
    float_,
    str_,
    list_,
    mrun,
):
    overrides = {
        "+none_": none_,
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
        assert job.return_value.none_ == none_
        assert job.return_value.int_ == int_
        assert job.return_value.bool_ == bool_
        assert job.return_value.float_ == float_
        assert job.return_value.str_ == str_
        assert job.return_value.list_ == list_
        assert job.return_value.mrun == mrun[i]


@pytest.mark.usefixtures("cleandir")
@settings(max_examples=10, deadline=None)
@given(
    none_=st.none(),
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
def test_overrides_kwargs_roundtrip(
    none_,
    int_,
    bool_,
    float_,
    str_,
    list_,
    mrun,
):
    overrides = {
        "+none_": none_,
        "+int_": int_,
        "+float_": float_,
        "+str_": str_,
        "+bool_": bool_,
        "+list_": hydra_list(list_),
        "+mrun": multirun(mrun),
    }
    jobs = launch(make_config(), instantiate, ["+foo=1"], multirun=True, **overrides)
    jobs1 = launch(make_config(), instantiate, {"+foo": 2}, multirun=True, **overrides)

    assert len(jobs[0]) == len(mrun)
    assert len(jobs1[0]) == len(mrun)
    for i, (job, job1) in enumerate(zip(jobs[0], jobs1[0])):
        assert job.return_value.foo == 1 and job1.return_value.foo == 2
        assert job.return_value.none_ == job1.return_value.none_ == none_
        assert job.return_value.int_ == job1.return_value.int_ == int_
        assert job.return_value.bool_ == job1.return_value.bool_ == bool_
        assert job.return_value.float_ == job1.return_value.float_ == float_
        assert job.return_value.str_ == job1.return_value.str_ == str_
        assert job.return_value.list_ == job1.return_value.list_ == list_
        assert job.return_value.mrun == job1.return_value.mrun == mrun[i]


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
def test_overrides_yaml(
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
        overrides_yamls = list(Path(job.working_dir).glob("**/overrides.yaml"))
        assert len(overrides_yamls) == 1
        overrides_yaml = overrides_yamls[0]

        yaml = load_from_yaml(overrides_yaml)
        assert isinstance(yaml, ListConfig)
        assert len(yaml) == 6
        for param_val in yaml:
            param, val = param_val.split("=")
            assert param in list(overrides.keys())

            override_val = overrides[param]
            if isinstance(override_val, multirun):
                assert val == str(mrun[i]).strip().replace(" ", "")
            else:
                assert val == str(override_val).strip().replace(" ", "")
