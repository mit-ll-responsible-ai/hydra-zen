# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import deque
from functools import partial
from inspect import signature
from typing import Any, Union

import pytest
from typing_extensions import Literal

from hydra_zen import (
    BuildsFn,
    ZenField,
    ZenStore,
    builds,
    instantiate,
    just,
    make_config,
    make_custom_builds_fn,
    to_yaml,
)
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError
from hydra_zen.typing import CustomConfigType, DataclassOptions
from hydra_zen.wrapper import default_to_config


class A:
    def __init__(self, x: Any) -> None:
        self.x = x

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, type(self)) and __value.x == self.x

    def __hash__(self) -> int:
        return hash(self.x) + hash(self.__class__.__name__)

    @staticmethod
    def static():
        return 11


class B: ...


class MyBuildsFn(BuildsFn[CustomConfigType[A]]):
    @classmethod
    def _make_hydra_compatible(
        cls,
        value: Any,
        allow_zen_conversion: bool = True,
        *,
        error_prefix: str = "",
        field_name: str = "",
        structured_conf_permitted: bool = True,
        convert_dataclass: bool,
        hydra_recursive: Union[bool, None] = None,
        hydra_convert: Union[Literal["none", "partial", "all", "object"], None] = None,
        zen_dataclass: Union[DataclassOptions, None] = None,
    ) -> Any:
        if isinstance(value, A):
            return cls.builds(A, value.x)

        return super()._make_hydra_compatible(
            value,
            allow_zen_conversion=allow_zen_conversion,
            error_prefix=error_prefix,
            field_name=field_name,
            structured_conf_permitted=structured_conf_permitted,
            convert_dataclass=convert_dataclass,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            zen_dataclass=zen_dataclass,
        )

    @classmethod
    def _sanitized_type(
        cls,
        type_: Any,
        *,
        primitive_only: bool = False,
        wrap_optional: bool = False,
        nested: bool = False,
    ) -> type:
        if type_ is B:
            return A

        return super()._sanitized_type(
            type_,
            primitive_only=primitive_only,
            wrap_optional=wrap_optional,
            nested=nested,
        )

    @classmethod
    def _get_obj_path(cls, target: Any) -> str:
        if target is A.static:
            return cls._get_obj_path(A) + ".static"
        return super()._get_obj_path(target)


def f(x):
    return x


def test_call():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        builds(dict, x=A(22))  # pyright: ignore

    assert instantiate(MyBuildsFn.builds(dict, x=A(22))) == {"x": A(22)}
    assert instantiate(MyBuildsFn.builds(dict, x=A(22), zen_partial=True))() == {
        "x": A(22)
    }
    assert instantiate(
        MyBuildsFn.builds(dict, x=[A(22), A(33)], zen_partial=True)
    )() == {"x": [A(22), A(33)]}


def test_just():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        just(A(22))

    assert instantiate(MyBuildsFn.just(A(22))) == A(22)
    assert instantiate(MyBuildsFn.just(dict(x=A(22)))) == {"x": A(22)}


def foo(x: A = A(5)):
    return x


def test_make_custom_builds():
    new_builds = make_custom_builds_fn(builds_fn=MyBuildsFn)
    assert instantiate(new_builds(dict, x=A(22))) == {"x": A(22)}

    new_fbuilds = make_custom_builds_fn(
        builds_fn=MyBuildsFn, populate_full_signature=True
    )
    assert instantiate(new_fbuilds(foo)().x) == A(5)

    new_pbuilds = make_custom_builds_fn(builds_fn=MyBuildsFn, zen_partial=True)
    assert instantiate(new_pbuilds(foo))() == A(5)


def bar(x: B): ...


def test_sanitized_type_override():
    # should swap B -> A in annotation
    out = signature(MyBuildsFn.builds(bar, populate_full_signature=True)).parameters[
        "x"
    ]
    assert out.annotation is A


def test_make_config():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        make_config(x=A(1))  # type: ignore

    assert instantiate(MyBuildsFn.make_config(x=A(1))().x) == A(1)


def test_zen_field():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        ZenField(default=A(1))
    ZenField(default=A(1), _builds_fn=MyBuildsFn)


def test_default_to_config():
    store = ZenStore("my store")(
        to_config=partial(default_to_config, CustomBuildsFn=MyBuildsFn)
    )
    store(A, x=A(x=2), name="blah")
    assert instantiate(store[None, "blah"]) == A(x=A(x=2))


@pytest.mark.parametrize(
    "obj",
    [
        deque([A(x=1), A(x=2)]),
        partial(foo, x=A(x=1)),
    ],
)
def test_zen_conversion_uses_custom_builds(obj):
    Conf = MyBuildsFn.just(obj)
    to_yaml(Conf)
    if not isinstance(obj, partial):
        assert instantiate(Conf) == obj
    else:
        assert instantiate(Conf)() == obj()


def test_parameterization_example():
    # just make sure that there are no runtime issues with CustomConfigType
    from hydra_zen import BuildsFn

    class MyType: ...

    class MyBuilds(BuildsFn[CustomConfigType[MyType]]): ...

    builds = MyBuilds.builds
    assert instantiate(builds(dict, x=1)) == dict(x=1)


def test_partial_supported():
    Cfg = MyBuildsFn.builds(dict, x=partial(A, x=2))
    to_yaml(Cfg)
    assert instantiate(Cfg)["x"]() == A(x=2)


# A simple subclass for testing purposes
class StandardBuilds(BuildsFn):
    pass

# Test case for a standard string
def test_make_hydra_compatible_with_string():
    """Tests that a standard string is returned as-is."""
    input_str = "hello world"
    assert StandardBuilds._make_hydra_compatible(input_str, convert_dataclass=False) == "hello world"