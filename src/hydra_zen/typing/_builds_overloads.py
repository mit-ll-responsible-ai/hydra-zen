# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# Stores overloads for `builds` with different default-values for signature
# pyright: strict

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

from typing_extensions import Literal, ParamSpec, Protocol

from ._implementations import (
    Builds,
    BuildsWithSig,
    DataClass_,
    Importable,
    PartialBuilds,
    SupportedPrimitive,
    ZenWrappers,
)

R = TypeVar("R")
P = ParamSpec("P")

__all__ = ["FullBuilds", "PBuilds"]


class FullBuilds(Protocol):  # pragma: no cover
    # full_builds -> populate_full_signature=True  (new default value)
    @overload
    def __call__(
        self,
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
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

    @overload
    def __call__(
        self,
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
    ) -> Type[Builds[Importable]]:
        ...

    @overload
    def __call__(
        self,
        hydra_target: Importable,
        *pos_args: SupportedPrimitive,
        zen_partial: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        populate_full_signature: Literal[True] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        **kwargs_for_target: SupportedPrimitive,
    ) -> Type[Builds[Importable]]:
        ...

    @overload
    def __call__(
        self,
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
    ) -> Type[PartialBuilds[Importable]]:
        ...

    @overload
    def __call__(
        self,
        *pos_args: Union[Importable, Callable[P, R], SupportedPrimitive],
        zen_partial: bool,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        populate_full_signature: bool,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        frozen: bool = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = (),
        dataclass_name: Optional[str] = ...,
        **kwargs_for_target: SupportedPrimitive,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...

    def __call__(
        self,
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
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...


class PBuilds(Protocol):  # pragma: no cover
    # partial_builds -> zen_partial=True (new default value)

    @overload
    def __call__(
        self,
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
    ) -> Type[PartialBuilds[Importable]]:
        ...

    @overload
    def __call__(
        self,
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
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

    @overload
    def __call__(
        self,
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
    ) -> Type[Builds[Importable]]:
        ...

    @overload
    def __call__(
        self,
        hydra_target: Importable,
        *pos_args: SupportedPrimitive,
        zen_partial: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        populate_full_signature: Literal[True] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        **kwargs_for_target: SupportedPrimitive,
    ) -> Type[Builds[Importable]]:
        ...

    @overload
    def __call__(
        self,
        *pos_args: Union[Importable, Callable[P, R], SupportedPrimitive],
        zen_partial: bool,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        populate_full_signature: bool = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        frozen: bool = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        **kwargs_for_target: SupportedPrimitive,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...

    def __call__(
        self,
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
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...


def f(x: PBuilds, y: bool, pop: bool):
    x(int, zen_partial=y, populate_full_signature=pop)
