# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from abc import ABC
from collections import UserList, deque, namedtuple
from typing import (
    Callable,
    Deque,
    List,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import hypothesis.strategies as st
import pytest
from hypothesis import assume, example, given, settings
from omegaconf.errors import GrammarParseError
from typing_extensions import Annotated, Final, Literal

from hydra_zen import builds, instantiate, to_yaml
from hydra_zen._utils.coerce import coerce_sequences


class MyNamedTuple(NamedTuple):
    x: int
    y: str


MyNamedTuple2 = namedtuple("MyNamedTuple2", ["x", "y", "z"])


class MyList(UserList):
    pass


def f(x):
    return x


NoneType = type(None)


def f_for_fuzz(x):
    return


class MyMetaClass(ABC):
    pass


class MyClass:
    pass


@example(int)
@example(str)
@example(bool)
@example(tuple)
@example(list)
@example(dict)
@example(set)
@example(deque)
@example(MyList)
@example(MyClass)
@example(MyMetaClass)
@example(Sequence[int])
@example(Tuple)
@example(Tuple[int, int])
@example(MyNamedTuple)
@example(Deque[str])
@example(List[int])
@example(Set[int])
@example(Callable[[int], int])
@example(Annotated[int, "meta"])
@example(Literal[1, 2])
@example(TypeVar("T"))
@example(NewType("NewInt", int))
@example(Final[str])
@example(Type[list])
@example(Optional[MyNamedTuple])
@example(Optional[int])
@example(Optional[Tuple[int, int]])
@example(Optional[Sequence[int]])
@example(Union[NoneType, Literal[1]])
@example(Union[Literal[1], NoneType])
@example(Union[Tuple[int, int], Tuple[int, int, int]])
@example(Union[Tuple[int, int], NoneType, Tuple[int, int, int]])
@given(st.from_type(type))
def test_fuzz_sequence_coercion(annotation):
    # test that `coerce_sequences` never raises
    f_for_fuzz.__annotations__.clear()
    f_for_fuzz.__annotations__["x"] = annotation
    coerce_sequences(f_for_fuzz)


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (int, int),
        (Annotated[int, "meta"], int),
        (Literal[1, 2], int),
        (str, str),
        (bool, bool),
        (Tuple, tuple),
        (Tuple[int, int], tuple),
        (Deque[str], deque),
        (List[int], list),
        (MyList, list),
        (Sequence[int], list),
        (MyNamedTuple, MyNamedTuple),
        (Optional[MyNamedTuple], (MyNamedTuple, NoneType)),
        (tuple, tuple),
        (deque, deque),
        (list, list),
        (Optional[int], (int, NoneType)),
        (Optional[Tuple[int, int]], (tuple, NoneType)),
        (Optional[Sequence[int]], (list, NoneType)),
        (Union[NoneType, Tuple[int, int]], (tuple, NoneType)),
        (Union[Tuple[int, int], NoneType], (tuple, NoneType)),
        (Union[Tuple[int, int], Tuple[int, int, int]], (list, NoneType)),
        (Union[Tuple[int, int], NoneType, Tuple[int, int, int]], (list, NoneType)),
    ],
)
def test_convert_sequences_against_many_types(in_type, expected_type):
    f.__annotations__["x"] = in_type

    @settings(max_examples=10)
    @given(x=st.from_type(in_type), as_named=st.booleans())
    def run(x, as_named: bool):
        args = ()
        kwargs = {}

        cast_x = list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else x

        if as_named:
            kwargs["x"] = cast_x
        else:
            args = (cast_x,)

        conf = builds(
            f, *args, **kwargs, hydra_convert="all", zen_wrappers=coerce_sequences
        )

        try:
            to_yaml(conf)  # ensure serializable
            out = instantiate(conf)
        except GrammarParseError:
            assume(False)  # generated string was bad-interp
            assert False  # unreachable

        assert isinstance(out, expected_type)

        if isinstance(expected_type, tuple):
            # case: Union
            caster, _ = expected_type
            if x is None:
                assert out is None
                return
        else:
            caster = expected_type

        if not isinstance(x, (MyNamedTuple, MyNamedTuple2)):
            assert out == caster(x)
        else:
            assert out == caster(*x)

    run()


def no_annotation(x, y=2, *args, z, **kwargs):
    ...


def no_sequences(x: int, y: bool, z: str):
    ...


def args_only(x, *args: tuple):
    ...


def kwargs_only(x, **kwargs: tuple):
    ...


@pytest.mark.parametrize(
    "target_fn", [no_annotation, no_sequences, args_only, kwargs_only]
)
def test_convert_sequences_no_annotations_is_noop(target_fn):
    assert target_fn is coerce_sequences(target_fn)


def test_convert_sequence_on_various_inputs():
    @coerce_sequences
    def f(x: MyNamedTuple, y: tuple, *args: tuple, z: Deque, **kwargs: tuple):
        return (x, y, *args, z) + tuple(kwargs.values())

    assert f(0, [1, 2], z=[3, 4]) == (0, (1, 2), deque([3, 4]))  # type: ignore
    assert f(0, y=[1, 2], z=[3, 4]) == (0, (1, 2), deque([3, 4]))  # type: ignore
    assert f(0, [1, 2], [-1], z=[3, 4]) == (0, (1, 2), [-1], deque([3, 4]))  # type: ignore
    assert f(0, [1, 2], [-1], z=[3, 4], extra=[5, 6]) == (  # type: ignore
        0,
        (1, 2),
        [-1],
        deque([3, 4]),
        [5, 6],
    )


def test_convert_sequences_on_class():
    @coerce_sequences
    class AClass:
        def __init__(self, x: tuple, y) -> None:
            self.x = x
            self.y = y

    out = AClass([1, 2], [3])  # type: ignore
    assert out.x == (1, 2)
    assert out.y == [3]


def test_invalid_type_annotation_doesnt_cause_internal_error():
    def f(x):
        return x

    f.__annotations__["x"] = 1  # an unexpected type hint
    assert coerce_sequences(f)(x=10) == 10
