# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from inspect import signature
from typing import Any, List, Union

import pytest
from typing_extensions import Literal

from hydra_zen import BuildsFn, builds, instantiate, just, make_custom_builds_fn
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError
from hydra_zen.typing import DataclassOptions, SupportedPrimitive
from hydra_zen.typing._implementations import DataclassOptions


class A:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, type(self)) and __value.x == self.x

    @staticmethod
    def static():
        return 11


class B:
    ...


class MyBuildsFn(
    BuildsFn[Union[SupportedPrimitive, A, List[Union[SupportedPrimitive, A]]]]
):
    @classmethod
    def _sanitized_default_value(
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
            return builds(A, value.x)

        return super()._sanitized_default_value(
            value,
            allow_zen_conversion,
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


my_builds = MyBuildsFn("my_builds")


def f(x):
    return x


def test_call():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        builds(dict, x=A(22))  # pyright: ignore

    assert instantiate(my_builds(dict, x=A(22))) == {"x": A(22)}
    assert instantiate(my_builds(dict, x=A(22), zen_partial=True))() == {"x": A(22)}
    assert instantiate(my_builds(dict, x=[A(22), A(33)], zen_partial=True))() == {
        "x": [A(22), A(33)]
    }


def test_just():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        just(A(22))

    assert instantiate(my_builds.just(A(22))) == A(22)
    assert instantiate(my_builds.just(dict(x=A(22)))) == {"x": A(22)}


def foo(x: A = A(5)):
    return x


def test_make_custom_builds():
    new_builds = make_custom_builds_fn(builds_fn=my_builds)
    assert instantiate(new_builds(dict, x=A(22))) == {"x": A(22)}

    new_fbuilds = make_custom_builds_fn(
        builds_fn=my_builds, populate_full_signature=True
    )
    assert instantiate(new_fbuilds(foo)().x) == A(5)

    new_pbuilds = make_custom_builds_fn(builds_fn=my_builds, zen_partial=True)
    assert instantiate(new_pbuilds(foo))() == A(5)


def bar(x: B):
    ...


def test_sanitized_type_override():
    # should swap B -> A in annotation
    out = signature(my_builds(bar, populate_full_signature=True)).parameters["x"]
    assert out.annotation is A


def test_get_obj_path_override():
    with pytest.raises(Exception, match="Error locating target"):
        assert instantiate(builds(A.static)) == 11
    assert instantiate(my_builds(A.static)) == 11
