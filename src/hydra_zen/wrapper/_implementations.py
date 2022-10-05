# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict

from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import hydra
from hydra.main import _UNSPECIFIED_  # type: ignore
from omegaconf import Container, DictConfig, ListConfig, OmegaConf
from typing_extensions import Literal, ParamSpec, TypeGuard

from hydra_zen import instantiate
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing._implementations import DataClass_

from .._compatibility import HYDRA_SUPPORTS_LIST_INSTANTIATION, SUPPORTS_VERSION_BASE
from ..structured_configs._type_guards import is_dataclass

__all__ = ["zen"]


T1 = TypeVar("T1")
P = ParamSpec("P")


if HYDRA_SUPPORTS_LIST_INSTANTIATION:
    _SUPPORTED_INSTANTIATION_TYPES: Tuple[Any, ...] = (dict, DictConfig, list, ListConfig)  # type: ignore
else:  # pragma: no cover
    _SUPPORTED_INSTANTIATION_TYPES: Tuple[Any, ...] = (dict, DictConfig)  # type: ignore


def is_instantiable(
    cfg: Any,
) -> TypeGuard[
    Union[
        DataClass_,
        Type[DataClass_],
        DictConfig,
        ListConfig,
        List[Any],
        Dict[str, Any],
    ]
]:
    return is_dataclass(cfg) or isinstance(cfg, _SUPPORTED_INSTANTIATION_TYPES)


SKIPPED_PARAM_KINDS = frozenset(
    (Parameter.POSITIONAL_ONLY, Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
)


PreCall = Optional[Union[Callable[[Any], Any], Iterable[Callable[[Any], Any]]]]


def _flat_call(x: Iterable[Callable[P, Any]]) -> Callable[P, None]:
    def f(*args: P.args, **kwargs: P.kwargs) -> None:
        for fn in x:
            fn(*args, **kwargs)

    return f


# TODO: enable specification of kwargs / extract-all
# TODO: Enable hydra_main to accept config directly -- auto register in config-store
# TODO: Enable kwarg-based calls of the function???
class Zen(Generic[P, T1]):
    # Specifies reserved parameter name specified to pass the
    # config through to the task function
    CFG_NAME: str = "zen_cfg"

    def __init__(
        self,
        func: Callable[P, T1],
        pre_call: PreCall = None,
    ) -> None:
        self.func: Callable[P, T1] = func
        try:
            self.parameters: Mapping[str, Parameter] = signature(self.func).parameters
        except (ValueError, TypeError):
            raise HydraZenValidationError(
                "hydra_zen.zen can only wrap callables that possess inspectable signatures."
            )

        if self.CFG_NAME in self.parameters:
            self._has_zen_cfg = True
            self.parameters = {
                name: param
                for name, param in self.parameters.items()
                if name != self.CFG_NAME
            }
        else:
            self._has_zen_cfg = False

        self._pre_call_iterable = (
            (pre_call,) if not isinstance(pre_call, Iterable) else pre_call
        )

        # validate pre-call signatures
        for _f in self._pre_call_iterable:
            if _f is None:
                continue

            _f_params = signature(_f).parameters

            if (sum(p.default is p.empty for p in _f_params.values()) > 1) or len(
                _f_params
            ) == 0:
                raise HydraZenValidationError(
                    f"pre_call function {_f} must be able to accept a single "
                    "positional argument"
                )

        self.pre_call: Optional[Callable[[Any], Any]] = (
            pre_call if not isinstance(pre_call, Iterable) else _flat_call(pre_call)
        )

    def _normalize_cfg(
        self,
        cfg: Union[
            DataClass_,
            Type[DataClass_],
            Dict[Any, Any],
            List[Any],
            ListConfig,
            DictConfig,
            str,
        ],
    ) -> Container:
        if is_dataclass(cfg):
            # ensures that default factories and interpolated fields
            # are resolved
            cfg = OmegaConf.structured(cfg)

        elif not OmegaConf.is_config(cfg):
            if not isinstance(cfg, (dict, list, str)):
                raise HydraZenValidationError(
                    f"`cfg` Must be a dataclass, dict/DictConfig, list/ListConfig, or "
                    f"yaml-string. Got {cfg}"
                )
            cfg = OmegaConf.create(cfg)

        assert isinstance(cfg, Container)
        return cfg

    def validate(self, __cfg: Any, excluded_params: Iterable[str] = ()) -> None:
        for _f in self._pre_call_iterable:
            if isinstance(_f, Zen):
                _f.validate(__cfg)

        cfg = self._normalize_cfg(__cfg)

        excluded_params = set(excluded_params)

        num_pos_only = sum(
            p.kind is p.POSITIONAL_ONLY for p in self.parameters.values()
        )

        _args_ = getattr(cfg, "_args_", [])

        if not isinstance(_args_, Sequence):
            raise HydraZenValidationError(
                f"`cfg._args_` must be a sequence type (e.g. a list), got {_args_}"
            )
        if num_pos_only and len(_args_) != num_pos_only:
            raise HydraZenValidationError(
                f"{self.func} has {num_pos_only} positional-only arguments, but "
                f"`cfg` specifies {len(getattr(cfg, '_args_', []))} positional "
                f"arguments via `_args_`."
            )

        missing_params: List[str] = []
        for name, param in self.parameters.items():
            if name in excluded_params:
                continue

            if param.kind in SKIPPED_PARAM_KINDS:
                continue

            if not hasattr(cfg, name) and param.default is param.empty:
                missing_params.append(name)

        if missing_params:
            raise HydraZenValidationError(
                f"`cfg` is missing the following fields: {', '.join(missing_params)}"
            )

    def __call__(
        self,
        __cfg: Union[
            DataClass_,
            Type[DataClass_],
            Dict[Any, Any],
            List[Any],
            ListConfig,
            DictConfig,
            str,
        ],
    ) -> T1:
        """
        Parameters
        ----------
        cfg : dict | list | DataClass | Type[DataClass] | str
            (positional only) A config object or yaml-string whose attributes will be
            extracted by-name according to the signature of `func` and passed to `func`.

            Attributes of types that can be instantiated by Hydra will be instantiated
            prior to being passed to `func`.
        """
        cfg = self._normalize_cfg(__cfg)

        # resolves all interpolated values in-place
        OmegaConf.resolve(cfg)

        if self.pre_call is not None:
            self.pre_call(cfg)

        args_ = list(getattr(cfg, "_args_", []))

        cfg_kwargs = {
            name: (
                getattr(cfg, name, param.default)
                if param.default is not param.empty
                else getattr(cfg, name)
            )
            for name, param in self.parameters.items()
            if param.kind not in SKIPPED_PARAM_KINDS
        }

        extra_kwargs = {self.CFG_NAME: cfg} if self._has_zen_cfg else {}

        return self.func(
            *(instantiate(x) if is_instantiable(x) else x for x in args_),
            **{
                name: instantiate(val) if is_instantiable(val) else val
                for name, val in cfg_kwargs.items()
            },
            **extra_kwargs,
        )  # type: ignore

    def hydra_main(
        self,
        config_path: Optional[str] = _UNSPECIFIED_,
        config_name: Optional[str] = None,
        version_base: Optional[str] = _UNSPECIFIED_,
    ) -> Callable[[Any], Any]:
        """
        Returns a Hydra-CLI compatible version of the wrapped function: `hydra.main(zen(func))`

        Parameters
        ----------
        config_path : Optional[str]
            The config path, a directory relative to the declaring python file.

            If config_path is not specified no directory is added to the Config search path.

        config_name : Optional[str]
            The name of the config (usually the file name without the .yaml extension)

        version_base : Optional[str]
            There are three classes of values that the version_base parameter supports, given new and existing users greater control of the default behaviors to use.

            - If the version_base parameter is not specified, Hydra 1.x will use defaults compatible with version 1.1. Also in this case, a warning is issued to indicate an explicit version_base is preferred.
            - If the version_base parameter is None, then the defaults are chosen for the current minor Hydra version. For example for Hydra 1.2, then would imply config_path=None and hydra.job.chdir=False.
            - If the version_base parameter is an explicit version string like "1.1", then the defaults appropriate to that version are used.

        Returns
        -------
        hydra_main : Callable[[Any], Any]
            hydra.main(zen(func))
        """

        kw = dict(config_path=config_path, config_name=config_name)
        if SUPPORTS_VERSION_BASE:
            kw["version_base"] = version_base

        return hydra.main(**kw)(self)()


@overload
def zen(
    __func: Callable[P, T1],
    *,
    pre_call: PreCall = ...,
    ZenWrapper: Type[Zen[P, T1]] = Zen,
) -> Zen[P, T1]:  # pragma: no cover
    ...


@overload
def zen(
    __func: Literal[None] = ...,
    *,
    pre_call: PreCall = ...,
    ZenWrapper: Type[Zen[Any, Any]] = ...,
) -> Callable[[Callable[P, T1]], Zen[P, T1]]:  # pragma: no cover
    ...


def zen(
    __func: Optional[Callable[P, T1]] = None,
    *,
    pre_call: PreCall = None,
    ZenWrapper: Type[Zen[P, T1]] = Zen,
) -> Union[Zen[P, T1], Callable[[Callable[P, T1]], Zen[P, T1]]]:

    if __func is not None:
        return ZenWrapper(__func, pre_call=pre_call)

    def wrap(f: Callable[P, T1]) -> Zen[P, T1]:
        return ZenWrapper(func=f, pre_call=pre_call)

    return wrap
