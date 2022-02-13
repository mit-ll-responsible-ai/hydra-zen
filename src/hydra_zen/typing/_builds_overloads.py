# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# Stores overloads for `builds` with different default-values for signature

from typing import Callable, Mapping, Optional, Tuple, Type, TypeVar, Union, overload

from typing_extensions import Literal, ParamSpec

from ._implementations import (
    Builds,
    Importable,
    PartialBuilds,
    SupportedPrimitive,
    ZenWrappers,
    _DataClass,
)

R = TypeVar("R")
P = ParamSpec("P")

__all__ = ["full_builds", "partial_builds"]


# full_builds -> populate_full_signature=True  (new default value)
@overload
def full_builds(
    hydra_target: Callable[P, R],
    *,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: Literal[True] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[()] = ...,
    frozen: bool = ...,
) -> Callable[P, Builds[Type[R]]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: Literal[False] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Union[Importable, Callable[P, R]],
    *pos_args: SupportedPrimitive,
    zen_partial: bool = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Callable[P, Builds[Type[R]]],
]:  # pragma: no cover
    ...


def full_builds(  # type: ignore
    *pos_args,
    zen_partial=False,
    zen_wrappers=tuple(),
    zen_meta=None,
    populate_full_signature=True,
    hydra_recursive=None,
    hydra_convert=None,
    frozen: bool = False,
    builds_bases=(),
    dataclass_name=None,
    **kwargs_for_target,
):  # pragma: no cover
    raise NotImplementedError()


# partial_builds -> zen_partial=True (new default value)


@overload
def partial_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Callable[P, R],
    *,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: Literal[True] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[()] = ...,
    frozen: bool = ...,
) -> Callable[P, Builds[Type[R]]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: Literal[False] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Union[Importable, Callable[P, R]],
    *pos_args: SupportedPrimitive,
    zen_partial: bool = ...,
    zen_wrappers: ZenWrappers = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[_DataClass], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Callable[P, Builds[Type[R]]],
]:  # pragma: no cover
    ...


def partial_builds(  # type: ignore
    *pos_args,
    zen_partial=True,
    zen_wrappers=tuple(),
    zen_meta=None,
    populate_full_signature=False,
    hydra_recursive=None,
    hydra_convert=None,
    frozen: bool = False,
    builds_bases=(),
    dataclass_name=None,
    **kwargs_for_target,
):  # pragma: no coer
    raise NotImplementedError()
