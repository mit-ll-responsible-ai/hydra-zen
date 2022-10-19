# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIR
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
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import hydra
from hydra.main import _UNSPECIFIED_  # type: ignore
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing_extensions import Literal, ParamSpec, TypeAlias, TypeGuard

from hydra_zen import instantiate
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing._implementations import DataClass_

from .._compatibility import HYDRA_SUPPORTS_LIST_INSTANTIATION, SUPPORTS_VERSION_BASE
from ..structured_configs._type_guards import is_dataclass
from ..structured_configs._utils import safe_name

__all__ = ["zen"]


R = TypeVar("R")
P = ParamSpec("P")


if HYDRA_SUPPORTS_LIST_INSTANTIATION:
    _SUPPORTED_INSTANTIATION_TYPES: Tuple[Any, ...] = (dict, DictConfig, list, ListConfig)  # type: ignore
else:  # pragma: no cover
    _SUPPORTED_INSTANTIATION_TYPES: Tuple[Any, ...] = (dict, DictConfig)  # type: ignore

ConfigLike: TypeAlias = Union[
    DataClass_,
    Type[DataClass_],
    Dict[Any, Any],
    DictConfig,
]


def is_instantiable(
    cfg: Any,
) -> TypeGuard[ConfigLike]:
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


class Zen(Generic[P, R]):
    """Implements the decorator logic that is exposed by `hydra_zen.zen`

    Attributes
    ----------
    CFG_NAME : str
        The reserved parameter name specifies to pass the input config through
        to the inner function. Can be overwritted via subclassing. Defaults
        to 'zen_cfg'

    See Also
    --------
    zen : A decorator that returns a function that will auto-extract, resolve, and instantiate fields from an input config based on the decorated function's signature.
    """

    # Specifies reserved parameter name specified to pass the
    # config through to the task function
    CFG_NAME: str = "zen_cfg"

    def __repr__(self) -> str:
        return f"zen[{(safe_name(self.func))}({', '.join(self.parameters)})](cfg, /)"

    def __init__(
        self,
        __func: Callable[P, R],
        *,
        exclude: Optional[Union[str, Iterable[str]]] = None,
        pre_call: PreCall = None,
        unpack_kwargs: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[Sig, R]
            The function being decorated. (This is a positional-only argument)

        unpack_kwargs: bool, optional (default=False)
            If `True` a `**kwargs` field in the decorated function's signature will be
            populated by all of the input config entries that are not specified by the
            rest of the signature (and that are not specified by the `exclude`
            argument).

        pre_call : Optional[Callable[[Any], Any] | Iterable[Callable[[Any], Any]]]
            One or more functions that will be called with the input config prior
            to the decorated functions. An iterable of pre-call functions are called
            from left (low-index) to right (high-index).

        exclude: Optional[str | Iterable[str]]
            Specifies one or more parameter names in the function's signature
            that will not be extracted from input configs by the zen-wrapped function.

            A single string of comma-separated names can be specified.
        """
        self.func: Callable[P, R] = __func

        try:
            self.parameters: Mapping[str, Parameter] = signature(self.func).parameters
        except (ValueError, TypeError):
            raise HydraZenValidationError(
                "hydra_zen.zen can only wrap callables that possess inspectable signatures."
            )

        if not isinstance(unpack_kwargs, bool):  # type: ignore
            raise TypeError(f"`unpack_kwargs` must be type `bool` got {unpack_kwargs}")

        self._unpack_kwargs: bool = unpack_kwargs and any(
            p.kind is p.VAR_KEYWORD for p in self.parameters.values()
        )

        self._exclude: Set[str]

        if exclude is None:
            self._exclude = set()
        elif isinstance(exclude, str):
            self._exclude = {k.strip() for k in exclude.split(",")}
        else:
            self._exclude = set(exclude)

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

    @staticmethod
    def _normalize_cfg(
        cfg: Union[
            DataClass_,
            Type[DataClass_],
            Dict[Any, Any],
            List[Any],
            ListConfig,
            DictConfig,
            str,
        ],
    ) -> DictConfig:
        if is_dataclass(cfg):
            # ensures that default factories and interpolated fields
            # are resolved
            cfg = OmegaConf.structured(cfg)

        elif not OmegaConf.is_config(cfg):
            if not isinstance(cfg, (dict, str)):
                raise HydraZenValidationError(
                    f"`cfg` must be a dataclass, dict/DictConfig, or "
                    f"dict-style yaml-string. Got {cfg}"
                )
            cfg = OmegaConf.create(cfg)

        if not isinstance(cfg, DictConfig):
            raise HydraZenValidationError(
                f"`cfg` must be a dataclass, dict/DictConfig, or "
                f"dict-style yaml-string. Got {cfg}"
            )
        return cfg

    def validate(self, __cfg: Union[ConfigLike, str]) -> None:
        """Validates the input config based on the decorated function without calling said function.

        Parameters
        ----------
        cfg : dict | list | DataClass | Type[DataClass] | str
            (positional only) A config object or yaml-string whose attributes will be
            checked according to the signature of `func`.

        Raises
        ------
        HydraValidationError
            `cfg` is not a valid input to the zen-wrapped function.
        """
        for _f in self._pre_call_iterable:
            if isinstance(_f, Zen):
                _f.validate(__cfg)

        cfg = self._normalize_cfg(__cfg)

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
            if name in self._exclude:
                continue

            if param.kind in SKIPPED_PARAM_KINDS:
                continue

            if not hasattr(cfg, name) and param.default is param.empty:
                missing_params.append(name)

        if missing_params:
            raise HydraZenValidationError(
                f"`cfg` is missing the following fields: {', '.join(missing_params)}"
            )

    def __call__(self, __cfg: Union[ConfigLike, str]) -> R:
        """
        Extracts values from the input config based on the decorated function's
        signature, resolves & instantiates them, and calls the function with them.

        Parameters
        ----------
        cfg : dict | DataClass | Type[DataClass] | str
            (positional only) A config object or yaml-string whose attributes will be
            extracted by-name according to the signature of `func` and passed to `func`.

            Attributes of types that can be instantiated by Hydra will be instantiated
            prior to being passed to `func`.

        Returns
        -------
        func_out : R
            The result of `func(<args extracted from cfg>)`
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
            if param.kind not in SKIPPED_PARAM_KINDS and name not in self._exclude
        }

        extra_kwargs = {self.CFG_NAME: cfg} if self._has_zen_cfg else {}
        if self._unpack_kwargs:
            names = (
                name
                for name in cfg
                if name not in cfg_kwargs
                and name not in self._exclude
                and isinstance(name, str)
            )
            cfg_kwargs.update({name: cfg[name] for name in names})
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
            Equivalent to `hydra.main(zen(func), [...])()`
        """

        kw = dict(config_path=config_path, config_name=config_name)
        if SUPPORTS_VERSION_BASE:
            kw["version_base"] = version_base

        return hydra.main(**kw)(self)()


@overload
def zen(
    __func: Callable[P, R],
    *,
    unpack_kwargs: bool = ...,
    pre_call: PreCall = ...,
    ZenWrapper: Type[Zen[P, R]] = Zen,
    exclude: Optional[Union[str, Iterable[str]]] = None,
) -> Zen[P, R]:  # pragma: no cover
    ...


@overload
def zen(
    __func: Literal[None] = ...,
    *,
    unpack_kwargs: bool = ...,
    pre_call: PreCall = ...,
    ZenWrapper: Type[Zen[Any, Any]] = ...,
    exclude: Optional[Union[str, Iterable[str]]] = None,
) -> Callable[[Callable[P, R]], Zen[P, R]]:  # pragma: no cover
    ...


def zen(
    __func: Optional[Callable[P, R]] = None,
    *,
    unpack_kwargs: bool = False,
    pre_call: PreCall = None,
    exclude: Optional[Union[str, Iterable[str]]] = None,
    ZenWrapper: Type[Zen[P, R]] = Zen,
) -> Union[Zen[P, R], Callable[[Callable[P, R]], Zen[P, R]]]:
    r"""zen(func, /, pre_call, ZenWrapper)

    A decorator that returns a function that will auto-extract, resolve, and
    instantiate fields from an input config based on the decorated function's signature.

    .. code-block:: pycon

       >>> Cfg = dict(x=1, y=builds(int, 4), z="${y}", unused=100)
       >>> zen(lambda x, y, z : x+y+z)(Cfg)  # x=1, y=4, z=4
       9

    The main purpose of `zen` is to enable a user to write/use Hydra-agnostic functions
    as the task functions for their Hydra app.

    Parameters
    ----------
    func : Callable[Sig, R]
        The function being decorated. (This is a positional-only argument)

    unpack_kwargs: bool, optional (default=False)
        If `True` a `**kwargs` field in the decorated function's signature will be
        populated by all of the input config entries that are not specified by the rest
        of the signature (and that are not specified by the `exclude` argument).

    pre_call : Optional[Callable[[Any], Any] | Iterable[Callable[[Any], Any]]]
        One or more functions that will be called with the input config prior
        to the decorated functions. An iterable of pre-call functions are called
        from left (low-index) to right (high-index).

        This is useful, e.g., for seeding a RNG prior to the instantiation phase
        that is triggered when calling the decorated function.

    exclude: Optional[str | Iterable[str]]
        Specifies one or more parameter names in the function's signature
        that will not be extracted from input configs by the zen-wrapped function.

        A single string of comma-separated names can be specified.

    ZenWrapper : Type[hydra_zen.wrapper.Zen], optional (default=Zen)
        If specified, a subclass of `Zen` that customizes the behavior of the wrapper.

    Returns
    -------
    Zen[Sig, R]
        A callable with signature `(conf: ConfigLike, \\) -> R`

        The wrapped function is an instance of `hydra_zen.wrapper.Zen` and accepts
        a single Hydra config (a dataclass, dictionary, or omegaconf container). The
        parameters of the decorated function's signature determine the fields that are
        extracted from the config; only those fields that are accessed will be resolved
        and instantiated.

    Notes
    -----
    ConfigLike is DataClass | list[Any] | dict[str, Any] | DictConfig | ListConfig

    The fields extracted from the input config are determined by the signature of the
    decorated function. There is an exception: including a parameter named "zen_cfg"
    in the function's signature will signal to `zen` to pass through the full config to
    that field (This specific parameter name can be overridden via `Zen.CFG_NAME`).

    All values (extracted from the input config) of types belonging to ConfigLike will
    be instantiated before being passed to the wrapped function.

    Examples
    --------
    **Basic Usage**

    Using `zen` as a decorator

    >>> from hydra_zen import zen, make_config, builds
    >>> @zen
    ... def f(x, y): return x + y

    The resulting decorated function accepts a single argument: a Hydra-compatible config that has the attributes "x" and "y":

    >>> f
    zen[f(x, y)](cfg, /)

    "Configs" – dataclasses, dictionaries, and omegaconf containers – are acceptable
    inputs to zen-wrapped functions. Interpolated fields will be resolved and
    sub-configs will be instantiated. Excess fields in the config are unused.

    >>> f(make_config(x=1, y=2, z=999))  # z is not used
    3
    >>> f(dict(x=2, y="${x}"))  # y will resolve to 2
    4
    >>> f(dict(x=2, y=builds(int, 10)))  # y will instantiate to 10
    12

    The wrapped function can be accessed directly

    >>> f.func
    <function __main__.f(x, y)>
    >>> f.func(-1, 1)
    0

    `zen` is compatible with partial'd functions.

    >>> from functools import partial
    >>> pf = partial(lambda x, y: x + y, x=10)
    >>> zpf = zen(pf)
    >>> zpf(dict(y=1))
    11
    >>> zpf(dict(x='${y}', y=1))
    2

    One can specify `exclude` to prevent particular variables from being extracted from
    a config:

    >>> def f(x=1, y=2): return (x, y)
    >>> zen(f)(dict(x=-10, y=-20))  # extracts x & y from config to call f
    (-10, -20)
    >>> zen(f, exclude="x")(dict(x=-10, y=-20))  # extracts y from config to call f(x=1, ...)
    (1, -20)
    >>> zen(f, exclude="x,y")(dict(x=-10, y=-20))  # defers to f's defaults
    (1, 2)

    Populating a `**kwargs` field via `unpack_kwargs=True`:

    >>> def f(a, **kw):
    ...     return a, kw

    >>> cfg = dict(a=1, b=22)
    >>> zen(f, unpack_kwargs=False)(cfg)
    (1, {})
    >>> zen(f, unpack_kwargs=True)(cfg)
    (1, {'b': 22})

    **Including a pre-call function**

    Given that a zen-wrapped function will automatically extract and instantiate config
    fields upon being called, it can be necessary to include a pre-call step that
    occurs prior to any instantiation. `zen` can be passed one or more pre-call
    functions that will be called with the input config as a precursor to calling the
    decorated function.

    Consider the following scenario where the config's instantiation involves drawing a
    random value, which we want to be made deterministic with a configurable seed. We
    will use a pre-call function to seed the RNG prior to the subsequent instantiation.

    >>> import random
    >>> from hydra_zen import builds, zen

    >>> Cfg = dict(
    ...         # `rand_val` will be instantiated and draw from randint upon
    ...         # calling the zen-wrapped function, thus we need a pre-call
    ...         # function to set the global RNG seed prior to instantiation
    ...         rand_val=builds(random.randint, 0, 10),
    ...         seed=0,
    ... )

    >>> @zen(pre_call=lambda cfg: random.seed(cfg.seed))
    ... def f(rand_val: int):
    ...     return rand_val

    >>> [f(Cfg) for _ in range(10)]
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


    **Using `@zen` instead of `@hydra.main`**

    The object returned by zen provides a convenience method – `Zen.hydra_main` – so
    that users need not double-wrap with `@hydra.main` to create a CLI:

    .. code-block:: python

        # example.py
        from hydra.core.config_store import ConfigStore

        from hydra_zen import builds, zen

        def task(x: int, y: int):
            print(x + y)

        cs = ConfigStore.instance()
        cs.store(name="my_app", node=builds(task, populate_full_signature=True))


        if __name__ == "__main__":
            zen(task).hydra_main(config_name="my_app", config_path=None)

    .. code-block:: console

       $ python example.py x=1 y=2
       3


    **Validating input configs**

    An input config can be validated against a zen-wrapped function without calling said function via the `.validate` method.

    >>> def f(x: int): ...
    >>> zen_f = zen(f)
    >>> zen_f.validate({"x": 1})  # OK
    >>> zen_f.validate({"y": 1})  # Missing x
    HydraZenValidationError: `cfg` is missing the following fields: x

    Validation propagates through zen-wrapped pre-call functions:

    >>> zen_f = zen(f, pre_call=zen(lambda seed: None))
    >>> zen_f.validate({"x": 1, "seed": 10})  # OK
    >>> zen_f.validate({"x": 1})  # Missing seed as required by pre-call
    HydraZenValidationError: `cfg` is missing the following fields: seed

    **Passing Through The Config**

    Some task functions require complete access to the full config to gain access to
    sub-configs. One can specify the field named `zen_config` in their task function's
    signature to signal `zen` that it should pass the full config to that parameter .

    >>> @zen
    ... def f(x: int, zen_cfg):
    ...     return x, zen_cfg
    >>> f(dict(x=1, y="${x}"))
    (1, {'x': 1, 'y': 1})
    """
    if __func is not None:
        return ZenWrapper(
            __func, pre_call=pre_call, exclude=exclude, unpack_kwargs=unpack_kwargs
        )

    def wrap(f: Callable[P, R]) -> Zen[P, R]:
        return ZenWrapper(
            f, pre_call=pre_call, exclude=exclude, unpack_kwargs=unpack_kwargs
        )

    return wrap
