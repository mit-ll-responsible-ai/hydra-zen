import inspect
import string

import hypothesis.strategies as st
import pytest
from hypothesis import assume, example, given

from hydra_zen import builds, make_custom_builds_fn, to_yaml
from tests.custom_strategies import partition, valid_builds_args

_builds_sig = inspect.signature(builds)
BUILDS_DEFAULTS = {
    name: p.default
    for name, p in _builds_sig.parameters.items()
    if p.kind is p.KEYWORD_ONLY
}
del _builds_sig
BUILDS_NAMES = set(BUILDS_DEFAULTS)


@example(args=[], kwargs=dict(__b=len))
@given(
    args=st.lists(st.none() | st.booleans()),
    kwargs=st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1).filter(
            lambda x: x not in BUILDS_DEFAULTS
        ),
        st.none() | st.booleans(),
    ),
)
def test_make_custom_builds_doesnt_accept_args_not_named_by_builds(args, kwargs):
    assume(args or kwargs)

    with pytest.raises(TypeError):
        # arbitrary args & kwargs
        make_custom_builds_fn(*args, **kwargs)


def f1(x: int):
    return


def f2(x, y: str):
    return


@pytest.mark.filterwarnings(
    "ignore:A structured config was supplied for `zen_wrappers`"
)
@given(
    kwargs=partition(valid_builds_args(), ordered=False),
    target=st.sampled_from([f1, f2]),
)
def test_make_builds_produces_builds_with_expected_config(kwargs, target):
    kwargs_as_defaults, kwargs_passed_through = kwargs

    # set a random partition of args as defaults to a custom builds
    custom_builds = make_custom_builds_fn(**kwargs_as_defaults)
    # pass the remainder of args directly to the customized builds
    via_custom = custom_builds(target, **kwargs_passed_through)

    # this should be the same as passing all of these args directly to vanilla builds
    via_builds = builds(target, **kwargs_passed_through, **kwargs_as_defaults)
    assert to_yaml(via_custom) == to_yaml(via_builds)


class BadGuy:
    pass


def _corrupt_kwargs(kwargs: dict):
    return {k: BadGuy for k in kwargs}


@given(
    bad_kwargs=valid_builds_args().map(_corrupt_kwargs),
)
def test_raises_on_bad_defaults(bad_kwargs):
    try:
        builds(f1, **bad_kwargs)
    except Exception as e:
        with pytest.raises(type(e)):
            make_custom_builds_fn(**bad_kwargs)
