# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# Stores overloads for `builds` with different default-values for signature
# pyright: strict
# pragma: no cover
from typing import (
    Any,
    Callable,
    Generic,
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
        __hydra_target: Type[BuildsWithSig[Type[R], P]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

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
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

    # partial=False, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[Builds[Importable]]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[Builds[Importable]]:
        ...

    # partial=True, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[PartialBuilds[Importable]]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[PartialBuilds[Importable]]:
        ...

    # partial=bool, pop-sig=False
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]],]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]],]:
        ...

    # partial=bool, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], Type[AnyBuilds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...

    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], Type[Builds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool] = None,
        populate_full_signature: bool = False,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        frozen: bool = False,
        builds_bases: Union[Tuple[Type[DataClass_], ...], Tuple[()]] = (),
        dataclass_name: Optional[str] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_convert: Optional[ZenConvert] = None,
        **kwargs_for_target: T,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:  # pragma: no cover
        ...


class FullBuilds(Generic[T]):
    # full_builds -> populate_full_signature=True  (new default value)

    # partial=False, pop_sig=True; no *args, **kwargs, nor builds_bases`

    @overload
    def __call__(
        self,
        __hydra_target: Type[BuildsWithSig[Type[R], P]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[Builds[Importable]]:
        ...

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
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

    @overload
    def __call__(
        self,
        __hydra_target: Type[Union[AnyBuilds[Importable], PartialBuilds[Importable]]],
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[Builds[Importable]]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[Builds[Importable]]:
        ...

    # partial=True, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[PartialBuilds[Importable]]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[PartialBuilds[Importable]]:
        ...

    # partial=bool, pop-sig=False
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        frozen: bool = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
        ...

    # partial=bool, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], Type[AnyBuilds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        frozen: bool = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        dataclass_name: Optional[str] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...

    def __call__(
        self,
        __hydra_target: Union[
            Callable[P, R],
            Importable,
            Type[Builds[Importable]],
            Type[PartialBuilds[Importable]],
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
        builds_bases: Union[Tuple[Type[DataClass_], ...], Tuple[()]] = (),
        dataclass_name: Optional[str] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_convert: Optional[ZenConvert] = None,
        **kwargs_for_target: T,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:  # pragma: no cover
        ...


class PBuilds(Generic[T]):
    # partial_builds -> zen_partial=True (new default value)

    # partial=True, pop-sig=bool
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[PartialBuilds[Importable]]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Type[PartialBuilds[Importable]]:
        ...

    # partial=False, pop-sig=True; no *args, **kwargs, nor builds_bases
    @overload
    def __call__(
        self,
        __hydra_target: Type[BuildsWithSig[Type[R], P]],
        *,
        zen_partial: Literal[False, None],
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

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
        builds_bases: Tuple[()] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> Type[BuildsWithSig[Type[R], P]]:
        ...

    # partial=bool, pop-sig=False
    @overload
    def __call__(
        self,
        __hydra_target: Type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]],]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]],]:
        ...

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
        builds_bases: Tuple[Type[DataClass_], ...] = ...,
        frozen: bool = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:
        ...

    def __call__(
        self,
        __hydra_target: Union[Callable[P, R], Type[AnyBuilds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool] = True,
        populate_full_signature: bool = False,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        frozen: bool = False,
        builds_bases: Union[Tuple[Type[DataClass_], ...], Tuple[()]] = (),
        dataclass_name: Optional[str] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_convert: Optional[ZenConvert] = None,
        **kwargs_for_target: T,
    ) -> Union[
        Type[Builds[Importable]],
        Type[PartialBuilds[Importable]],
        Type[BuildsWithSig[Type[R], P]],
    ]:  # pragma: no cover
        ...
