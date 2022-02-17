# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# Stores overloads for `builds` with different default-values for signature

from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Literal, ParamSpec

from ._implementations import (
    Builds,
    DataClass_,
    Importable,
    PartialBuilds,
    SupportedPrimitive,
    ZenWrappers,
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
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
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
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: Literal[False] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


@overload
def full_builds(
    hydra_target: Union[Importable, Callable[P, R]],
    *pos_args: SupportedPrimitive,
    zen_partial: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Callable[P, Builds[Type[R]]],
]:  # pragma: no cover
    ...


def full_builds(
    *pos_args: Union[Importable, Callable[P, R], SupportedPrimitive],
    zen_partial: bool = False,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = True,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    builds_bases: Tuple[Type[DataClass_], ...] = (),
    dataclass_name: Optional[str] = None,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Callable[P, Builds[Type[R]]],
]:  # pragma: no cover
    raise NotImplementedError()


# partial_builds -> zen_partial=True (new default value)


@overload
def partial_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Callable[P, R],
    *,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
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
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: Literal[False] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


@overload
def partial_builds(
    hydra_target: Union[Importable, Callable[P, R]],
    *pos_args: SupportedPrimitive,
    zen_partial: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    frozen: bool = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Callable[P, Builds[Type[R]]],
]:  # pragma: no cover
    ...


def partial_builds(
    *pos_args: Union[Importable, Callable[P, R], SupportedPrimitive],
    zen_partial: bool = True,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    builds_bases: Tuple[Type[DataClass_], ...] = (),
    dataclass_name: Optional[str] = None,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Callable[P, Builds[Type[R]]],
]:  # pragma: no cover
    raise NotImplementedError()
