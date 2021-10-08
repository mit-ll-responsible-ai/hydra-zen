# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# flake8: noqa

from collections import deque
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from omegaconf import OmegaConf
from typing_extensions import Annotated, Final, Literal, TypedDict

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


def a_func(x):
    return x


class UserIdentity(TypedDict):
    name: str
    surname: str


@skip_if_no_validators
@pytest.mark.parametrize(
    "annotation, fools_hydra",
    [
        (Tuple[str, int, bool], (True, "hi", 2)),
        (Dict[str, int], dict(a="hi")),
        (Literal["a", "b"], "c"),
        (Union[List[str], str], dict(a=1)),
        (MyNamedTuple, (1.0, 2.0)),
        (UserIdentity, dict(first="bruce", last="lee")),
        (Annotated[int, "special"], "hello"),
    ],
)
@pytest.mark.parametrize("validator", all_validators)
@settings(max_examples=20)
@given(data=st.data(), as_yaml=st.booleans())
def test_validations_missed_by_hydra(
    annotation, fools_hydra, validator, as_yaml: bool, data: st.DataObject
):
    """Tests variety of annotations not supported by Hydra.

    Ensures:
    - inputs that would be missed by Hydra get validated by provided validator
    - validation works post yaml-roundtrip
    - validated targets produce expected outputs
    - validation works in end-to-end instantiation workflows
    """

    # draw valid input
    valid_input = data.draw(st.from_type(annotation), label="valid_input")

    # boilerplate: set annotation of target to-be-built
    a_func.__annotations__.clear()
    a_func.__annotations__["x"] = annotation

    conf_no_val = builds(a_func, populate_full_signature=True, hydra_convert="all")

    conf_with_val = builds(
        a_func,
        populate_full_signature=True,
        zen_wrappers=validator,
        hydra_convert="all",
    )

    if as_yaml:
        conf_with_val = OmegaConf.create(to_yaml(conf_with_val))

    # Hydra misses bad input
    instantiate(conf_no_val, x=fools_hydra)

    with pytest.raises(TypeError):
        # validation catches bad input
        instantiate(conf_with_val, x=fools_hydra)

    # passes validation
    out = instantiate(conf_with_val, x=valid_input)

    assert isinstance(out, type(valid_input))
    assert out == valid_input


def func_with_sig(x: str, *args: int, y: Optional[Tuple[int, str]] = None, **kwargs):
    return (x, *args), {"y": y, **kwargs}


class ClassWithSig:
    def __init__(
        self, x: str, *args: int, y: Optional[Tuple[int, str]] = None, **kwargs
    ):
        self.out = (x, *args), {"y": y, **kwargs}


@skip_if_no_validators
@pytest.mark.parametrize("validator", all_validators)
@pytest.mark.parametrize(
    "args, kwargs, should_pass",
    # fmt: off
    [
        (("a",), {}, True),                      #                       f("a") ✓
        ((), dict(x="a"), True),                 #                     f(x="a") ✓
        (("a", 1, 2), {}, True),                 #                 f("a", 1, 2) ✓
        (("a",), dict(y=(1, "a")), True),        #          f("hi", y=(1, "a")) ✓
        (("a",), dict(y=[1, "a"]), True),        #          f("hi", y=[1, "a"]) ✓
        (("a",), dict(y=(1, "a"), z=1), True),   #      f("a", y=(1, "a"), z=1) ✓
        (("a",), dict(y=(1, "a"), z="a"), True), #    f("a", y=(1, "a"), z="a") ✓
        (([1],), {}, False),                     #                       f([1]) ✗
        ((), dict(x=[1]), False),                #                     f(x=[1]) ✗
        (("a", 2, "b"), {}, False),              #               f("a", 1, "b") ✗
        (("a",), dict(y=(1, None)), False),      #     f("a", y=(1, None), z=1) ✗
        (([1],), dict(y=(1, "a")), False),       #      f([1], y=(1, "a"), z=1) ✗
        (([1],), dict(y=(1, None)), False),      #     f([1], y=(1, None), z=1) ✗
    ],
    # fmt: on
)
@given(target=st.sampled_from([func_with_sig, ClassWithSig]), as_yaml=st.booleans())
def test_signature_parsing(args, kwargs, should_pass, validator, target, as_yaml):
    """Ensures validators handle various *args and **kwargs arrangements properly.

    Includes:
    - positional args
    - named args
    - annotated *args
    - kwd-only args"""
    conf_with_val = builds(
        target,
        *args,
        **kwargs,
        populate_full_signature=True,
        zen_wrappers=validator,
        hydra_convert="all",
    )
    if as_yaml:
        conf_with_val = OmegaConf.create(to_yaml(conf_with_val))
    if not should_pass:
        with pytest.raises(TypeError):
            instantiate(conf_with_val)
        return
    instantiate(conf_with_val)
