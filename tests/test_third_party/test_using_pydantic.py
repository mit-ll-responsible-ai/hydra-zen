import pytest
from pydantic import AnyUrl, PositiveFloat
from typing_extensions import Annotated

from hydra_zen.experimental.third_party.pydantic import validates_with_pydantic


@pytest.mark.parametrize(
    "custom_type, good_val, bad_val",
    [
        (PositiveFloat, 22, -1),
        (AnyUrl, "http://www.pythonlikeyoumeanit.com", "hello"),
    ],
)
def test_pydantic_specific_fields(custom_type, good_val, bad_val):
    def f(x):
        pass

    f.__annotations__["x"] = custom_type
    bear_hugged_f = validates_with_pydantic(f)

    bear_hugged_f(good_val)  # ok
    with pytest.raises(Exception):
        bear_hugged_f(bad_val)

    class A:
        def __init__(self, x) -> None:
            pass

    A.__init__.__annotations__["x"] = custom_type
    validates_with_pydantic(A)  # type: ignore

    A(good_val)
    with pytest.raises(Exception):
        A(bad_val)
