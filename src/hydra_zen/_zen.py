# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict
from dataclasses import is_dataclass
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from omegaconf import DictConfig, ListConfig, OmegaConf
from typing_extensions import Literal, ParamSpec, TypeGuard

from hydra_zen import instantiate
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing._implementations import DataClass

__all__ = ["zen"]


T1 = TypeVar("T1")
P = ParamSpec("P")


def is_config(cfg: Any) -> TypeGuard[Union[DataClass, DictConfig, ListConfig]]:
    return is_dataclass(cfg) or OmegaConf.is_config(cfg)


SKIPPED_PARAM_KINDS = frozenset(
    (Parameter.POSITIONAL_ONLY, Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
)


PreCall = Optional[Union[Callable[[Any], Any], Iterable[Callable[[Any], Any]]]]


def _flat_call(x: Iterable[Callable[P, Any]]) -> Callable[P, None]:
    def f(*args: P.args, **kwargs: P.kwargs) -> None:
        for fn in x:
            fn(*args, **kwargs)

    return f


# TODO: add hydra-main method
class Zen(Generic[P, T1]):
    def __init__(
        self,
        func: Callable[P, T1],
        pre_call: PreCall = None,
    ) -> None:
        self.func: Callable[P, T1] = func
        try:
            self.parameters = signature(self.func).parameters
        except (ValueError, TypeError):
            raise HydraZenValidationError(
                "hydra_zen.zen can only wrap callables that possess inspectable signatures."
            )
        self.pre_call = (
            pre_call if not isinstance(pre_call, Iterable) else _flat_call(pre_call)
        )

    def validate(self, cfg: Any, excluded_params: Iterable[str] = ()):
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

    def __call__(self, cfg: Any) -> T1:
        if callable(cfg):
            cfg = cfg()  # instantiate dataclass to resolve default-factory

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

        out = self.func(
            *(instantiate(x) if is_config(x) else x for x in args_),
            **{
                name: instantiate(val) if is_config(val) else val
                for name, val in cfg_kwargs.items()
            },
        )  # type: ignore

        return out


@overload
def zen(
    __func: Callable[P, T1],
    *,
    pre_call: PreCall = ...,
) -> Zen[P, T1]:  # pragma: no cover
    ...


@overload
def zen(
    __func: Literal[None] = ...,
    *,
    pre_call: PreCall = ...,
) -> Callable[[Callable[P, T1]], Zen[P, T1]]:  # pragma: no cover
    ...


def zen(
    __func: Optional[Callable[P, T1]] = None,
    *,
    pre_call: PreCall = None,
) -> Union[Zen[P, T1], Callable[[Callable[P, T1]], Zen[P, T1]]]:

    if __func is not None:
        return Zen(__func, pre_call=pre_call)

    def wrap(f: Callable[P, T1]) -> Zen[P, T1]:
        return Zen(func=f, pre_call=pre_call)

    return wrap