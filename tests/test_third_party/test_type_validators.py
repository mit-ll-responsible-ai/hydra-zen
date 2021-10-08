# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import deque
from typing import Deque, List, NamedTuple, Sequence, Tuple

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf
from typing_extensions import Final

from hydra_zen import builds, instantiate, to_yaml

all_validators = []


try:
    from hydra_zen.experimental.third_party.pydantic import validates_with_pydantic

    all_validators.append(validates_with_pydantic)
    del validates_with_pydantic
except ImportError:
    pass


skip_if_no_validators: Final = pytest.mark.skipif(
    not all_validators, reason="no thirdparty validators available"
)


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
@given(num_compositions=st.integers(0, 4), x=st.integers(), as_named_arg=st.booleans())
def test_multiple_wraps_on_func(
    validator, num_compositions: int, x: int, as_named_arg: bool
):
    def f(x: int):
        return x

    wrapped_f = f
    for _ in range(num_compositions):
        wrapped_f = validator(wrapped_f)

    assert f([1, 2]) == [1, 2]  # type: ignore

    if num_compositions:
        with pytest.raises(Exception):
            wrapped_f([1, 2])  # type: ignore
    if as_named_arg:
        assert f(x=x) == wrapped_f(x=x)
    else:
        assert f(x) == wrapped_f(x)


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
@given(num_compositions=st.integers(0, 4), x=st.integers(), as_named_arg=st.booleans())
def test_multiple_wraps_on_class(
    validator, num_compositions: int, x: int, as_named_arg: bool
):
    class A:
        def __init__(self, x: int):
            self.x = x

    wrapped_A = A
    for _ in range(num_compositions):
        wrapped_A = validator(wrapped_A)

    assert wrapped_A is A, "decoration should occur in-place"

    if num_compositions:
        with pytest.raises(Exception):
            wrapped_A([1, 2])  # type: ignore

    a_instance = wrapped_A(x=x) if as_named_arg else wrapped_A(x)
    assert isinstance(a_instance, A)
    assert a_instance.x == x


class MyNamedTuple(NamedTuple):
    x: float
    y: float
    z: float


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
@pytest.mark.parametrize(
    "annotation, caster",
    [
        (Sequence, list),
        (List[float], list),
        (List, list),
        (list, list),
        (Tuple[float, float, float], tuple),
        (Tuple, tuple),
        (tuple, tuple),
        (Deque[float], deque),
        (Deque, deque),
        (deque, deque),
        (MyNamedTuple, MyNamedTuple),
    ],
)
@given(
    x=st.lists(st.floats(allow_nan=False), min_size=3, max_size=3),
)
def test_coerces_lists_to_annotated_sequence_type(
    validator, x: List[float], annotation: type, caster: type
):
    def f(x):
        return x

    f.__annotations__["x"] = annotation

    wrapped_f = validator(f)
    coerced_out = wrapped_f(x)
    assert isinstance(f(x), list)  # type: ignore
    assert isinstance(coerced_out, caster)
    if caster is not MyNamedTuple:
        assert coerced_out == caster(x)
    else:
        assert coerced_out == caster(*x)


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
def test_on_custom_types(validator):
    class A:
        pass

    class B:
        pass

    @validator
    def f(x: Tuple[A, B]):
        return x

    a = A()
    b = B()

    with pytest.raises(Exception):
        f((a, a))  # type: ignore

    assert f((a, b)) == (a, b)


def run_with_hydra(x: float):
    return x


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
@pytest.mark.parametrize("as_yaml", [True, False])
def test_hydra_compatible_func_target(validator, as_yaml: bool):

    conf_with_val = builds(
        run_with_hydra, populate_full_signature=True, zen_wrappers=validator
    )
    if as_yaml:
        conf_with_val = OmegaConf.create(to_yaml(conf_with_val))

    with pytest.raises(Exception):
        instantiate(conf_with_val, x="not a float")

    out = instantiate(conf_with_val, x=1)
    assert isinstance(out, float)
    assert out == 1.0


class RunWithHydra:
    def __init__(self, x: float):
        self.x = x


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
@pytest.mark.parametrize("as_yaml", [True, False])
def test_hydra_compatible_class_target(validator, as_yaml: bool):
    conf_with_val = builds(
        RunWithHydra, populate_full_signature=True, zen_wrappers=validator
    )

    if as_yaml:
        conf_with_val = OmegaConf.create(to_yaml(conf_with_val))

    with pytest.raises(Exception):
        instantiate(conf_with_val, x="not a float")

    out = instantiate(conf_with_val, x=1)
    assert isinstance(out, RunWithHydra)
    assert isinstance(out.x, float)
    assert out.x == 1.0
