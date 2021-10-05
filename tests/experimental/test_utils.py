from collections import UserList, deque, namedtuple
from typing import Deque, List, NamedTuple, Optional, Sequence, Tuple, Union

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings
from omegaconf.errors import GrammarParseError

from hydra_zen import builds, instantiate, to_yaml
from hydra_zen.experimental.utils import convert_sequences


class MyNamedTuple(NamedTuple):
    x: int
    y: str


MyNamedTuple2 = namedtuple("MyNamedTuple2", ["x", "y", "z"])


class MyList(UserList):
    pass


def f(x):
    return x


NoneType = type(None)


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (int, int),
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
            f, *args, **kwargs, hydra_convert="all", zen_wrappers=convert_sequences
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
    assert target_fn is convert_sequences(target_fn)


def test_convert_sequence_on_various_inputs():
    @convert_sequences
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
    @convert_sequences
    class AClass:
        def __init__(self, x: tuple, y) -> None:
            self.x = x
            self.y = y

    out = AClass([1, 2], [3])  # type: ignore
    assert out.x == (1, 2)
    assert out.y == [3]
