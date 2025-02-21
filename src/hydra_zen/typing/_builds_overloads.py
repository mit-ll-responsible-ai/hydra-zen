# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from collections.abc import Mapping

# Stores overloads for `builds` with different default-values for signature
# pyright: strict
# pragma: no cover
from typing import Any, Callable, Generic, Optional, TypeVar, Union, overload

from typing_extensions import Literal, ParamSpec

from ._implementations import (
    AnyBuilds,
    Builds,
    BuildsWithSig,
    DataClass_,
    DataclassOptions,
    DefaultsList,
    PartialBuilds,
    SupportedPrimitive,
    ZenConvert,
    ZenWrappers,
)

Importable = TypeVar("Importable", bound=Callable[..., Any])
T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")

__all__ = ["FullBuilds", "PBuilds", "StdBuilds"]


class StdBuilds(Generic[T]):
    # partial=False, pop-sig=True; no *args, **kwargs, nor builds_bases
    @overload
    def __call__(
        self,
        __hydra_target: type[BuildsWithSig[type[R], P]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    # partial=False, pop-sig=True; no *args, **kwargs, nor builds_bases
    @overload
    def __call__(
        self,
        __hydra_target: Callable[P, R],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    # partial=False, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[Builds[Importable]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[Builds[Importable]]: ...

    # partial=True, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[True],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    # partial=bool, pop-sig=False
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
    ]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
    ]: ...

    # partial=bool, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], type[AnyBuilds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]: ...

    def __call__(
        self,
        __hydra_target: Union[
            Callable[P, R],
            type[Builds[Importable]],
            Importable,
            type[BuildsWithSig[type[R], P]],
        ],
        *pos_args: T,
        zen_partial: Optional[bool] = None,
        populate_full_signature: bool = False,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        frozen: bool = False,
        builds_bases: Union[tuple[type[DataClass_], ...], tuple[()]] = (),
        dataclass_name: Optional[str] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_convert: Optional[ZenConvert] = None,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]:  # pragma: no cover
        ...


class FullBuilds(Generic[T]):
    # full_builds -> populate_full_signature=True  (new default value)

    # partial=False, pop_sig=True; no *args, **kwargs, nor builds_bases`

    @overload
    def __call__(
        self,
        __hydra_target: type[BuildsWithSig[type[R], P]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[Builds[Importable]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Callable[P, R],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: type[Union[AnyBuilds[Importable], PartialBuilds[Importable]]],
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[Builds[Importable]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[Builds[Importable]]: ...

    # partial=True, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[True],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    # partial=bool, pop-sig=False
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        frozen: bool = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[type[Builds[Importable]], type[PartialBuilds[Importable]]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        frozen: bool = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[type[Builds[Importable]], type[PartialBuilds[Importable]]]: ...

    # partial=bool, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], type[AnyBuilds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        frozen: bool = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]: ...

    def __call__(
        self,
        __hydra_target: Union[
            Callable[P, R],
            Importable,
            type[Builds[Importable]],
            type[PartialBuilds[Importable]],
            type[BuildsWithSig[type[R], P]],
        ],
        *pos_args: T,
        zen_partial: Optional[bool] = None,
        populate_full_signature: bool = True,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        frozen: bool = False,
        builds_bases: Union[tuple[type[DataClass_], ...], tuple[()]] = (),
        dataclass_name: Optional[str] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_convert: Optional[ZenConvert] = None,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]:  # pragma: no cover
        ...


class PBuilds(Generic[T]):
    # partial_builds -> zen_partial=True (new default value)

    # partial=True, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[True] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    # partial=False, pop-sig=True; no *args, **kwargs, nor builds_bases
    @overload
    def __call__(
        self,
        __hydra_target: type[BuildsWithSig[type[R], P]],
        *,
        zen_partial: Literal[False, None],
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Callable[P, R],
        *,
        zen_partial: Literal[False, None],
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    # partial=bool, pop-sig=False
    @overload
    def __call__(
        self,
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
    ]: ...

    @overload
    def __call__(
        self,
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
    ]: ...

    # partial=bool, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], Importable],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: bool,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]: ...

    def __call__(
        self,
        __hydra_target: Union[
            Callable[P, R],
            type[AnyBuilds[Importable]],
            Importable,
            type[BuildsWithSig[type[R], P]],
        ],
        *pos_args: T,
        zen_partial: Optional[bool] = True,
        populate_full_signature: bool = False,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        frozen: bool = False,
        builds_bases: Union[tuple[type[DataClass_], ...], tuple[()]] = (),
        dataclass_name: Optional[str] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_convert: Optional[ZenConvert] = None,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]:  # pragma: no cover
        ...
