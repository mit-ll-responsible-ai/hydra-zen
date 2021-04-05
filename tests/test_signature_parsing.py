import inspect
from inspect import Parameter
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from hydra_utils import builds, hydrated_dataclass, mutable_value
from tests import valid_hydra_literals

Empty = Parameter.empty


@pytest.mark.parametrize("as_hydrated_dataclass", [False, True])
@given(user_value=valid_hydra_literals, full_signature=st.booleans())
def test_user_specified_value_overrides_default(
    user_value, as_hydrated_dataclass: bool, full_signature: bool
):
    def f(x=2):
        return x

    if not as_hydrated_dataclass:
        BuildsF = builds(f, x=user_value, populate_full_signature=full_signature)
    else:

        @hydrated_dataclass(f, populate_full_signature=full_signature)
        class BuildsF:
            x: Any = (
                mutable_value(user_value)
                if isinstance(user_value, (list, dict))
                else user_value
            )

    b = BuildsF()
    assert b.x == user_value


@settings(max_examples=1000)
@given(
    user_value_x=valid_hydra_literals,
    user_value_y=valid_hydra_literals,
    user_value_z=valid_hydra_literals,
    default_value=valid_hydra_literals,
    specified_as_default=st.lists(st.sampled_from(["x", "y", "z"]), unique=True),
)
def test_builds_signature_shuffling_takes_least_path(
    user_value_x, user_value_y, user_value_z, default_value, specified_as_default
):

    # We will specify an arbitrary selection of x, y, z via `builds`, and then specify the
    # remaining parameters via initializing the resulting dataclass. This ensures that we can
    # accommodate arbitrary "signature shuffling", i.e. that parameters with defaults specified
    # are shuffled just to the right of those without defaults.
    #
    # E.g.
    #  - `builds(f, populate_full_signature=True)`.__init__ -> (x, y, z, has_default=default_value)
    #  - `builds(f, x=1, populate_full_signature=True)`.__init__ -> (y, z, x=1, has_default=default_value)
    #  - `builds(f, y=2, z=-1, populate_full_signature=True)`.__init__ -> (z, y=2, z=-1, has_default=default_value)
    def f(x, y, z, has_default=default_value):
        return x, y, z, has_default

    defaults = dict(x=user_value_x, y=user_value_y, z=user_value_z)

    default_override = {k: defaults[k] for k in specified_as_default}
    specified_via_init = {
        k: defaults[k] for k in set(defaults) - set(specified_as_default)
    }

    BuildsF = builds(f, **default_override, populate_full_signature=True)
    sig_param_names = [p.name for p in inspect.signature(BuildsF).parameters.values()]
    expected_param_ordering = (
        sorted(specified_via_init) + sorted(specified_as_default) + ["has_default"]
    )

    assert sig_param_names == expected_param_ordering

    b = BuildsF(**specified_via_init)
    assert b.x == user_value_x
    assert b.y == user_value_y
    assert b.z == user_value_z
    assert b.has_default == default_value


@pytest.mark.parametrize("include_extra_param", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_builds_with_full_sig_mirrors_target_sig(
    include_extra_param: bool, partial: bool
):
    def target(x: str, *args, y: int = 22, z=[2], **kwargs):
        pass

    kwargs = dict(named_param=2) if include_extra_param else {}
    kwargs["y"] = 0  # overwrite default value
    Conf = builds(target, populate_full_signature=True, hydra_partial=partial, **kwargs)

    params = inspect.signature(Conf).parameters.values()

    expected_sig = [("x", str), ("y", int), ("z", Any)]
    if include_extra_param:
        expected_sig.append(("named_param", Any))

    actual_sig = [(p.name, p.annotation) for p in params]
    assert expected_sig == actual_sig

    conf = Conf(x=-100)
    assert conf.x == -100
    assert conf.y == 0
    assert conf.z == [2]

    if include_extra_param:
        assert conf.named_param == 2
