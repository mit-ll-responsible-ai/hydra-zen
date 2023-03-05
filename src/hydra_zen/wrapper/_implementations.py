# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict

import warnings
from collections import defaultdict, deque
from functools import wraps
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    Generator,
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
    cast,
    overload,
)

import hydra
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing_extensions import (
    Final,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    TypeGuard,
)

from hydra_zen import instantiate, just, make_custom_builds_fn
from hydra_zen._compatibility import HYDRA_VERSION, Version
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.structured_configs._type_guards import safe_getattr
from hydra_zen.structured_configs._utils import get_obj_path
from hydra_zen.typing._implementations import (
    DataClass_,
    GroupName,
    Node,
    NodeName,
    StoreEntry,
)

from .._compatibility import HYDRA_SUPPORTS_LIST_INSTANTIATION, SUPPORTS_VERSION_BASE
from ..structured_configs._type_guards import is_dataclass
from ..structured_configs._utils import safe_name

__all__ = ["zen", "store", "Zen"]


R = TypeVar("R")
P = ParamSpec("P")
F = TypeVar("F")


_UNSPECIFIED_: Any = object()

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
    """Implements the wrapping logic that is exposed by `hydra_zen.zen`

    Attributes
    ----------
    func : Callable[Sig, R]
        The function that was wrapped.

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
        func : Callable[Sig, R], positional-only
            The function being wrapped.

        unpack_kwargs: bool, optional (default=False)
            If `True` a `**kwargs` field in the wrapped function's signature will be
            populated by all of the input config entries that are not specified by the
            rest of the signature (and that are not specified by the `exclude`
            argument).

        pre_call : Optional[Callable[[Any], Any] | Iterable[Callable[[Any], Any]]]
            One or more functions that will be called with the input config prior
            to the wrapped functions. An iterable of pre-call functions are called
            from left (low-index) to right (high-index).

        exclude: Optional[str | Iterable[str]]
            Specifies one or more parameter names in the function's signature
            that will not be extracted from input configs by the zen-wrapped function.

            A single string of comma-separated names can be specified.
        """
        self.func: Callable[P, R] = __func

        try:
            # Must cast to dict so that `self` is pickle-compatible.
            self.parameters: Mapping[str, Parameter] = dict(
                signature(self.func).parameters
            )
        except (ValueError, TypeError):
            raise HydraZenValidationError(
                "hydra_zen.zen can only wrap callables that possess inspectable "
                "signatures."
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

    # TODO: add "extract" option that enables returning dict of fields
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
                safe_getattr(cfg, name, param.default)
                if param.default is not param.empty
                else safe_getattr(cfg, name)
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
        Generates a Hydra-CLI for the wrapped function. Equivalent to `hydra.main(zen(func), [...])()`

        Parameters
        ----------
        config_path : Optional[str]
            The config path, an absolute path to a directory or a directory relative to
            the declaring python file. If `config_path` is not specified no directory is
            added to the config search path.

            Specifying `config_path` via `Zen.hydra_main` is only supported for
            Hydra 1.3.0+.

        config_name : Optional[str]
            The name of the config (usually the file name without the .yaml extension)

        version_base : Optional[str]
            There are three classes of values that the version_base parameter supports,
            given new and existing users greater control of the default behaviors to
            use.

            - If the version_base parameter is not specified, Hydra 1.x will use defaults compatible with version 1.1. Also in this case, a warning is issued to indicate an explicit version_base is preferred.
            - If the version_base parameter is None, then the defaults are chosen for the current minor Hydra version. For example for Hydra 1.2, then would imply config_path=None and hydra.job.chdir=False.
            - If the version_base parameter is an explicit version string like "1.1", then the defaults appropriate to that version are used.

        Returns
        -------
        hydra_main : Callable[[Any], Any]
            Equivalent to `hydra.main(zen(func), [...])()`
        """

        kw = dict(config_name=config_name)

        # For relative config paths, Hydra looks in the directory relative to the file
        # in which the task function is defined. Unfortunately, it is only able to
        # follow wrappers starting in Hydra 1.3.0. Thus `Zen.hydra_main` cannot
        # handle string config_path entries until Hydra 1.3.0
        if (config_path is _UNSPECIFIED_ and HYDRA_VERSION < Version(1, 2, 0)) or (
            (
                isinstance(config_path, str)
                or (config_path is _UNSPECIFIED_ and version_base == "1.1")
            )
            and HYDRA_VERSION < Version(1, 3, 0)
        ):  # pragma: no cover
            warnings.warn(
                "Specifying config_path via hydra_zen.zen(...).hydra_main "
                "is only supported for Hydra 1.3.0+"
            )
        if Version(1, 3, 0) <= HYDRA_VERSION and isinstance(config_path, str):
            # Here we create an on-the-fly wrapper so that Hydra can trace
            # back through the wrapper to the original task function
            # We could give `Zen` as `__wrapped__` attr, but this messes with
            # things like `inspect.signature`.
            #
            # A downside of this is that `wrapper` is not pickle-able.
            @wraps(self.func)
            def wrapper(cfg: Any):
                return self(cfg)

            target = wrapper
        else:
            target = self

        if config_path is not _UNSPECIFIED_:
            kw["config_path"] = config_path

        if (
            SUPPORTS_VERSION_BASE and version_base is not _UNSPECIFIED_
        ):  # pragma: no cover
            kw["version_base"] = version_base

        return hydra.main(**kw)(target)()


@overload
def zen(
    __func: Callable[P, R],
    *,
    unpack_kwargs: bool = ...,
    pre_call: PreCall = ...,
    ZenWrapper: Type[Zen[P, R]] = Zen,
    exclude: Optional[Union[str, Iterable[str]]] = None,
) -> Zen[P, R]:
    ...


@overload
def zen(
    __func: Literal[None] = ...,
    *,
    unpack_kwargs: bool = ...,
    pre_call: PreCall = ...,
    ZenWrapper: Type[Zen[Any, Any]] = ...,
    exclude: Optional[Union[str, Iterable[str]]] = None,
) -> Callable[[Callable[P, R]], Zen[P, R]]:
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

    A wrapper that returns a function that will auto-extract, resolve, and
    instantiate fields from an input config based on the wrapped function's signature.

    .. code-block:: pycon

       >>> fn = lambda x, y, z : x+y+z
       >>> wrapped_fn = zen(fn)

       >>> cfg = dict(x=1, y=builds(int, 4), z="${y}", unused=100)
       >>> wrapped_fn(cfg)  # x=1, y=4, z=4
       9

    The main purpose of `zen` is to enable a user to write/use Hydra-agnostic functions
    as the task functions for their Hydra app. See "Notes" for more details.

    Parameters
    ----------
    func : Callable[Sig, R], positional-only
        The function being wrapped.

    unpack_kwargs: bool, optional (default=False)
        If `True` a `**kwargs` field in the wrapped function's signature will be
        populated by all of the input config entries that are not specified by the rest
        of the signature (and that are not specified by the `exclude` argument).

    pre_call : Optional[Callable[[Any], Any] | Iterable[Callable[[Any], Any]]]
        One or more functions that will be called with the input config prior
        to the wrapped function. An iterable of pre-call functions are called
        from left (low-index) to right (high-index).

        This is useful, e.g., for seeding a RNG prior to the instantiation phase
        that is triggered when calling the wrapped function.

    exclude: Optional[str | Iterable[str]]
        Specifies one or more parameter names in the function's signature
        that will not be extracted from input configs by the zen-wrapped function.

        A single string of comma-separated names can be specified.

    ZenWrapper : Type[hydra_zen.wrapper.Zen], optional (default=Zen)
        If specified, a subclass of `Zen` that customizes the behavior of the wrapper.

    Returns
    -------
    wrapped : Zen[Sig, R]
        A callable with signature `(conf: ConfigLike, \\) -> R`

        The wrapped function is an instance of `hydra_zen.wrapper.Zen` and accepts
        a single Hydra config (a dataclass, dictionary, or omegaconf container). The
        parameters of the wrapped function's signature determine the fields that are
        extracted from the config; only those fields that are accessed will be resolved
        and instantiated.

    See Also
    --------
    hydra_zen.wrapper.Zen : Implements the wrapping logic that is exposed by `hydra_zen.zen`.

    Notes
    -----
    The following pseudo code conveys the core functionality of `zen`:

    .. code-block:: python

       from hydra_zen import instantiate as inst

       def zen(func):
           sig = get_signature(func)

           def wrapped(cfg):
               cfg = resolve_interpolated_fields(cfg)
               kwargs = {p: inst(getattr(cfg, p)) for p in sig}
               return func(**kwargs)
           return wrapped

    The presence of a parameter named "zen_cfg" in the wrapped function's signature
    will cause `zen` to pass the full, resolved config to that field. This specific
    parameter name can be overridden via `Zen.CFG_NAME`.

    Specifying `config_path` via `Zen.hydra_main` is only supported for Hydra 1.3.0+.

    Examples
    --------
    **Basic Usage**

    >>> from hydra_zen import zen, make_config, builds
    >>> def f(x, y): return x + y
    >>> zen_f = zen(f)

    The wrapped function – `zen_f` – accepts a single argument: a Hydra-compatible
    config that has the attributes "x" and "y":

    >>> zen_f
    zen[f(x, y)](cfg, /)

    "Configs" – dataclasses, dictionaries, and omegaconf containers – are acceptable
    inputs to zen-wrapped functions. Interpolated fields will be resolved and
    sub-configs will be instantiated. Excess fields in the config are unused.

    >>> zen_f(make_config(x=1, y=2, z=999))  # z is not used
    3
    >>> zen_f(dict(x=2, y="${x}"))  # y will resolve to 2
    4
    >>> zen_f(dict(x=2, y=builds(int, 10)))  # y will instantiate to 10
    12

    The wrapped function can be accessed directly

    >>> zen_f.func
    <function __main__.f(x, y)>
    >>> zen_f.func(-1, 1)
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

    >>> def g(x=1, y=2): return (x, y)
    >>> cfg = {"x": -10, "y": -20}
    >>> zen(g)(cfg)  # extracts x & y from config to call f
    (-10, -20)
    >>> zen(g, exclude="x")(cfg)  # extracts y from config to call f(x=1, ...)
    (1, -20)
    >>> zen(g, exclude="x,y")(cfg)  # defers to f's defaults
    (1, 2)

    Populating a `**kwargs` field via `unpack_kwargs=True`:

    >>> def h(a, **kw):
    ...     return a, kw

    >>> cfg = dict(a=1, b=22)
    >>> zen(h, unpack_kwargs=False)(cfg)
    (1, {})
    >>> zen(h, unpack_kwargs=True)(cfg)
    (1, {'b': 22})


    **Passing Through The Full Input Config**

    Some task functions require complete access to the full config to gain access to
    sub-configs. One can specify the field named `zen_config` in their task function's
    signature to signal `zen` that it should pass the full config to that parameter .

    >>> @zen
    ... def zf(x: int, zen_cfg):
    ...     return x, zen_cfg
    >>> zf(dict(x=1, y="${x}", foo="bar"))
    (1, {'x': 1, 'y': 1, 'foo': 'bar'})

    **Including a pre-call function**

    Given that a zen-wrapped function will automatically extract and instantiate config
    fields upon being called, it can be necessary to include a pre-call step that
    occurs prior to any instantiation. `zen` can be passed one or more pre-call
    functions that will be called with the input config as a precursor to calling the
    decorated function.

    Consider the following scenario where the instantiating the input config involves
    drawing a random value, which we want to be made deterministic with a configurable
    seed. We will use a pre-call function to seed the RNG prior to the instantiation.

    >>> import random
    >>> from hydra_zen import builds, zen
    >>>
    >>> def func(rand_val: int): return rand_val
    >>>
    >>> cfg = dict(
    ...         seed=0,
    ...         rand_val=builds(random.randint, 0, 10),
    ... )
    >>> wrapped = zen(func, pre_call=lambda cfg: random.seed(cfg.seed))

    >>> @zen(pre_call=lambda cfg: random.seed(cfg.seed))
    ... def f1(rand_val: int):
    ...     return rand_val

    >>> [f1(cfg) for _ in range(10)]
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


    **Using `zen` instead of `@hydra.main`**

    The object returned by zen provides a convenience method – `Zen.hydra_main` –
    to generate a CLI for a zen-wrapped task function:

    .. code-block:: python

       # example.py
       from hydra_zen import zen, store

       @store(name="my_app")
       def task(x: int, y: int):
           print(x + y)

       if __name__ == "__main__":
           store.add_to_hydra_store()
           zen(task).hydra_main(config_name="my_app", config_path=None, version_base="1.2")


    .. code-block:: console

       $ python example.py x=1 y=2
       3


    **Validating input configs**

    An input config can be validated against a zen-wrapped function – without calling
    said function – via the `.validate` method.

    >>> def f2(x: int): ...
    >>> zen_f = zen(f2)
    >>> zen_f.validate({"x": 1})  # OK
    >>> zen_f.validate({"y": 1})  # Missing x
    HydraZenValidationError: `cfg` is missing the following fields: x

    Validation propagates through zen-wrapped pre-call functions:

    >>> zen_f2 = zen(f2, pre_call=zen(lambda seed: None))
    >>> zen_f2.validate({"x": 1, "seed": 10})  # OK
    >>> zen_f2.validate({"x": 1})  # Missing seed as required by pre-call
    HydraZenValidationError: `cfg` is missing the following fields: seed
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


fbuilds = make_custom_builds_fn(populate_full_signature=True)


def default_to_config(
    target: Union[
        Callable[..., Any],
        DataClass_,
        List[Any],
        Dict[Any, Any],
        ListConfig,
        DictConfig,
    ],
    **kw: Any,
) -> Union[DataClass_, Type[DataClass_], ListConfig, DictConfig]:
    """Creates a config that describes `target`.

    This function is designed to selectively apply `hydra_zen.builds` or
    `hydra_zen.just` in a way that permits maximum compatibility with common
    inputs to `hydra_zen.ZenStore`. It behavior can be summarized based on the type of
    `target`

    - OmegaConf containers and dataclass *instances* are returned unchanged
    - A dataclass type is processed as `builds(target, **kw, populate_full_signature=True, builds_bases=(target,))`
    - Lists and dictionaries are processed by `hydra_zen.just`
    - All other inputs are processed as `builds(target, **kw, populate_full_signature=True)`

    Parameters
    ----------
    target : Callable[..., Any] | DataClass | Type[DataClass] | list | dict

    **kw : Any
        Keyword arguments to be passed to `builds`.

    Returns
    -------
    target_config :  DataClass | Type[DataClass] | list | dict

    Examples
    --------
    Lists and dictionaries

    >>> from hydra_zen.wrapper import default_to_config
    >>> default_to_config([1, {"z": 2+2j}])
    [1, {'z': ConfigComplex(real=2.0, imag=2.0, _target_='builtins.complex')}]

    Dataclass types

    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    ... class A:
    ...     x: int
    ...     y: int

    >>> Builds_A = default_to_config(A, y=22)
    >>> Builds_A(x=1)
    Builds_A(x=1, y=22, _target_='__main__.A')
    >>> issubclass(Builds_A, A)
    True

    A function

    >>> from hydra_zen import to_yaml
    >>> def func(x: int, y: int): ...
    >>> print(to_yaml(default_to_config(func)))
    _target_: __main__.func
    x: ???
    'y': ???
    """
    if is_dataclass(target):
        if isinstance(target, type):
            if issubclass(target, HydraConf):
                # don't auto-config HydraConf
                return target

            if not kw and get_obj_path(target).startswith("types."):
                # handles dataclasses returned by make_config()
                return target
            return fbuilds(target, **kw, builds_bases=(target,))
        if kw:
            raise ValueError(
                "store(<dataclass-instance>, [...]) does not support specifying "
                "keyword arguments"
            )
        return target

    elif isinstance(target, (dict, list)):
        # TODO: convert to OmegaConf containers?
        return just(target)
    elif isinstance(target, (DictConfig, ListConfig)):
        return target
    else:
        t = cast(Callable[..., Any], target)
        return fbuilds(t, **kw)


class _HasName(Protocol):
    __name__: str


# TODO: Should we automatically snake-case?
def get_name(target: _HasName) -> str:
    name = getattr(target, "__name__", None)
    if not isinstance(name, str):
        raise TypeError(
            f"Cannot infer config store entry name for {target}. It does not have a "
            f"`__name__` attribute. Please manually specify `store({target}, "
            f"name=<some name>, [...])`"
        )
    return name


class _StoreCallSig(TypedDict):
    """Arguments for ZenStore.__call__

    This default dict enables us to easily update/merge the default arguments for a
    specific ZenStore instance, in support of self-partialing behavior."""

    name: Union[NodeName, Callable[[Any], NodeName]]
    group: Union[GroupName, Callable[[Any], GroupName]]
    package: Optional[Union[str, Callable[[Any], str]]]
    provider: Optional[str]
    __kw: Dict[str, Any]  # kwargs passed to to_config
    to_config: Callable[[Any], Any]


# TODO: make frozen dict
defaults: Final = _StoreCallSig(
    name=get_name,
    group=None,
    package=None,
    provider=None,
    to_config=default_to_config,
    __kw={},
)

_DEFAULT_KEYS: Final[FrozenSet[str]] = frozenset(
    _StoreCallSig.__required_keys__ - {"__kw"}
)


class _Deferred:
    __slots__ = ("to_config", "target", "kw")

    def __init__(
        self, to_config: Callable[[F], Node], target: F, kw: Dict[str, Any]
    ) -> None:
        self.to_config = to_config
        self.target = target
        self.kw = kw

    def __call__(self) -> Any:
        return self.to_config(self.target, **self.kw)


def _resolve_node(entry: StoreEntry, copy: bool) -> StoreEntry:
    """Given an entry, updates the entry so that its node is not deferred, and returns
    the entry. This function is a passthrough for an entry whose node is not deferred"""
    item = entry["node"]
    if isinstance(item, _Deferred):
        entry["node"] = item()

    if copy:
        entry = entry.copy()
    return entry


class ZenStore:
    """An abstraction over Hydra's config store, which enables users to maintain
    multiple, isolated store instances before populating Hydra's global config store.

    `ZenStore` is also designed to consolidate the config-creation and storage process;
    it can be used to decorate config-targets and dataclasses, enabling "inline" config
    creation and storage patterns. This is also a "self-partialing" object, meaning a
    store instance can overwrite its own default values. Please consult the
    :ref:`examples <self-partial>` for more details.

    `hydra_zen.store` is available as a pre-instantiated, globally-available store,
    which is initialized as:

    .. code-block:: python

       store = ZenStore(
           name="zen_store",
           deferred_to_config=True,
           deferred_hydra_store=True,
       )

    Notes
    -----
    Special support is provided for overriding Hydra's configuration; the name and
    group of the store entry is inferred to be 'config' and 'hydra', respectively,
    when an instance/subclass of `HydraConf` is being stored. E.g., specifying

    .. code-block:: python

       from hydra.conf import HydraConf, JobConf
       from hydra_zen import store

       store(HydraConf(job=JobConf(chdir=True)))

    is equivalent to writing the following manually

    .. code-block:: python

       store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra", provider="hydra_zen")

    Additionally, overwriting the store entry for `HydraConf` will not raise an error
    even if `ZenStore(overwrite_ok=False)` is specified.

    Examples
    --------
    >>> from hydra_zen import to_yaml, store, ZenStore
    >>> def pyaml(x):
    ...     # for pretty printing configs
    ...     print(to_yaml(x))

    **Basic usage**

    Let's add a config to hydra-zen's pre-instantiated `ZenStore` instance. Each store
    entry must have an associated name. Optionally, a group, package, and/or provider
    may be specified for the entry as well.

    >>> config1 = {'name': 'Roger', 'age': 24}
    >>> config2 = {'name': 'Rita', 'age': 27}
    >>> _ = store(config1, name="roger", group="profiles")
    >>> _ = store(config2, name="rita", group="profiles")
    >>> store
    zen_store
    {'profiles': ['roger', 'rita']}

    A store's entries are keyed by their `(group, name)` pairs (the default group is
    `None`).

    >>> store["profiles", "roger"]  # (group, name) -> config node
    {'name': 'Roger', age: 24}

    By default, the stored config(s) will be "enqueued" for addition to Hydra's config
    store. The method `.add_to_hydra_store()` must be called to add the enqueued
    configs to Hydra's central store.

    >>> store.has_enqueued()
    True
    >>> store.add_to_hydra_store()  # adds all enqueued entries to Hydra's global store
    >>> store.has_enqueued()
    False

    By default, attempting to overwrite an entry will result in an error.

    >>> store({}, name="rita", group="profiles")  # same name and group as above
    ValueError: (name=rita group=profiles): Hydra config store entry already exists.
    Specify `overwrite_ok=True` to enable replacing config store entries

    We can create a distinct store that has an independent internal repository of
    configs.

    >>> new_store = ZenStore("new_store")
    >>> _ = new_store([1, 2, 3], name="backbone")

    >>> store
    zen_store
    {'profiles': ['roger', 'rita']}
    >>> new_store
    new_store
    {None: ['backbone']}

    .. _store-autoconf:

    **Auto-config capabilities**

    The input to a store is processed by the store's `to_config` function prior to
    creating the stored config node. This defaults to
    `hydra_zen.wrapper.default_to_config`, which applies `hydra_zen.builds` or
    `hydra_zen.just` to inputs based on their types.

    For instance, consider the following function:

    >>> def sum_it(a: int, b: int): return a + b

    We can pass `sum_it` directly to our store to leverage auto-config and auto-naming
    capabilities. Here, `builds(sum_it, a=1, b=2)` will be called under the hood by
    `new_store` to create the config for `sum_it`.

    >>> store2 = ZenStore()
    >>> _ = store2(sum_it, a=1, b=2)  # entry name defaults to `sum_it.__name__`
    >>> config = store2[None, "sum_it"]
    >>> pyaml(config)
    _target_: __main__.sum_it
    a: 1
    b: 2

    Refer to `hydra_zen.wrapper.default_to_config` for more details about the default
    auto-config behaviors of `ZenStore`.

    **Support for decorator patterns**

    `ZenStore.__call__` is a pass-through and can be used as a decorator. Let's add two
    store entries for `func` by decorating it.

    >>> store = ZenStore()

    >>> @store(a=1, b=22, name="func1")
    ... @store(a=-10, name="func2")
    ... def func(a: int, b: int):
    ...     return a - b

    Each application of `@store` utilizes the store's auto-config capability
    to create and store a config inline. I.e. the above snippet is equivalent to

    >>> from hydra_zen import builds
    >>>
    >>> store(builds(func, a=1, b=22), name="func1")
    >>> store(builds(func, a=-10,
    ...              populate_full_signature=True
    ...              ),
    ...       name="func2",
    ... )


    >>> func(10, 3)  # the decorated function is left unchanged
    7
    >>> pyaml(store[None, "func1"])
    _target_: __main__.func
    a: 1
    b: 22
    >>> pyaml(store[None, "func2"])
    _target_: __main__.func
    b: ???
    a: -10

    Note that, by default, the application of `to_config` via the store **is deferred
    until that entry is actually accessed**. This offsets the runtime cost of
    constructing configs for the decorated function so that it need not be paid until
    the config is actually accessed by the store.

    .. _self-partial:

    **Customizable store defaults via 'self-partialing' patterns**

    The default values for a store's `__call__` parameters – `group`, `to_config`, etc.
    – can easily be customized. Simply call the store with those new values and
    without specifying an object to be stored. This will return a "mirrored" store
    instance – with the same internal state as the original store – with updated
    defaults.

    For example, let's create a store where we want to store multiple configs under a
    `'math'` group and under a `'functools'` group, respectively.

    >>> import math
    >>> import functools

    >>> new_store = ZenStore()
    >>> math_store = new_store(group="math")  # overwrites group default
    >>> tool_store = new_store(group="functools")  # overwrites group default

    `math_store` and `tool_store` both share the same internal state as `new_store`, but
    have overwritten default values for the `group`.

    >>> math_store(math.floor)  # equivalent to: `new_store(math.floor, group="math")`
    >>> math_store(math.ceil)
    >>> tool_store(functools.lru_cache)
    >>> tool_store(functools.wraps)

    See that `new_store` has entries under these corresponding groups:

    >>> new_store
    custom_store
    {'math': ['floor', 'ceil'], 'functools': ['lru_cache', 'wraps']}

    These "self-partialing" patterns can be chained indefinitely and can be used to set
    partial defaults for the `to_config` function.

    >>> profile_store = new_store(group="profile")
    >>> schemaless = profile_store(schema="<none>")

    >>> from dataclasses import dataclass
    >>> @profile_store(name="admin", has_root=True)
    >>> @schemaless(name="test_admin", has_root=True)
    >>> @schemaless(name="test_user", has_root=False)
    >>> @dataclass
    >>> class Profile:
    >>>     username: str
    >>>     schema: str
    >>>     has_root: bool

    >>> pyaml(new_store["profile", "admin"])
    username: ???
    schema: ???
    has_root: true
    _target_: __main__.Profile

    >>> pyaml(new_store["profile", "test_admin"])
    username: ???
    schema: <none>
    has_root: true
    _target_: __main__.Profile
    """

    __slots__ = (
        "name",
        "_internal_repo",
        "_defaults",
        "_queue",
        "_deferred_to_config",
        "_deferred_store",
        "_overwrite_ok",
        "_warn_node_kwarg",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        deferred_to_config: bool = True,
        deferred_hydra_store: bool = True,
        overwrite_ok: bool = False,
        warn_node_kwarg: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        name : Optional[str]
            The name for this store.

        deferred_to_config : bool, default=True
            If `True` (default), this store will a not apply `to_config` to the
            target until that specific entry is accessed by the store.

        deferred_hydra_store : bool, default=True
            If `True` (default), this store will not add entries to Hydra's global
            config store until `store.add_to_hydra_store` is called explicitly.

        overwrite_ok : bool, default=False
            If `False` (default), attempting to overwrite entries in this store and
            trying to use this store to overwrite entries in Hydra's global store
            will raise a `ValueError`.

        warn_node_kwarg: bool, default=True
            If `True` specifying a `node` kwarg in `ZenStore.__call__` will emit a
            warning.

            This helps to protect users from mistakenly self-partializing a store
            with `store(node=Config)` instead of actually storing the node with
            `store(Config)`.
        """
        if not isinstance(deferred_to_config, bool):  # type: ignore
            raise TypeError(
                f"deferred_to_config must be a bool, got {deferred_to_config}"
            )

        if not isinstance(overwrite_ok, bool):  # type: ignore
            raise TypeError(f"overwrite_ok must be a bool, got {overwrite_ok}")

        if not isinstance(deferred_hydra_store, bool):  # type: ignore
            raise TypeError(
                f"deferred_hydra_store must be a bool, got {deferred_hydra_store}"
            )

        self.name: str = "custom_store" if name is None else name

        # The following attributes are mirrored across store instances that are
        # created via the 'self-partialing' process
        self._internal_repo: Dict[Tuple[GroupName, NodeName], StoreEntry] = {}
        # Internal repo entries that have yet to be added to Hydra's config store
        self._queue: Deque[StoreEntry] = deque([])

        self._deferred_to_config = deferred_to_config
        self._deferred_store = deferred_hydra_store
        self._overwrite_ok = overwrite_ok
        # Contains the current default arguments for `self.__call__`
        self._defaults: _StoreCallSig = defaults.copy()

        self._warn_node_kwarg = warn_node_kwarg

    def __repr__(self) -> str:
        # TODO: nicer repr?
        groups_contents: DefaultDict[Optional[str], List[str]] = defaultdict(list)
        for grp, name in self._internal_repo:
            groups_contents[grp].append(name)

        return f"{self.name}\n{repr(dict(groups_contents))}"

    def __eq__(self, __o: object) -> bool:
        """Returns `True` if two stores share identical internal repos and queues.

        Examples
        --------
        >>> from hydra_zen import ZenStore
        >>> store1 = ZenStore()
        >>> store2 = ZenStore()
        >>> store1_a = store1(group='a')
        >>> _ = store1_a(dict(x=1), name="foo")

        >>> store1 == store1_a
        True
        >>> store1 == store2
        False
        """
        if not isinstance(__o, ZenStore):
            return False
        return __o._internal_repo is self._internal_repo and __o._queue is self._queue

    # TODO: support *to_config_pos_args
    @overload
    def __call__(
        self,
        __target: F,
        *,
        name: Union[NodeName, Callable[[F], NodeName]] = ...,
        group: Union[GroupName, Callable[[F], GroupName]] = ...,
        package: Optional[Union[str, Callable[[F], str]]] = ...,
        provider: Optional[str] = ...,
        to_config: Callable[[F], Node] = default_to_config,
        **to_config_kw: Any,
    ) -> F:
        ...

    # TODO: ZenStore -> Self when mypy adds support for Self
    #       https://github.com/python/mypy/pull/11666
    @overload
    def __call__(
        self: Self,
        __target: Literal[None] = None,
        *,
        name: Union[NodeName, Callable[[Any], NodeName]] = ...,
        group: Union[GroupName, Callable[[Any], GroupName]] = ...,
        package: Optional[Union[str, Callable[[Any], str]]] = ...,
        provider: Optional[str] = ...,
        to_config: Callable[[Any], Node] = ...,
        **to_config_kw: Any,
    ) -> Self:
        ...

    def __call__(self, __target: Optional[F] = None, **kw: Any) -> Union[F, "ZenStore"]:
        """__call__(target : Optional[T] = None, /, name: NodeName | Callable[[Any], NodeName]] = ..., group: GroupName | Callable[[T], GroupName]] = None, package: Optional[str | Callable[[T], str]]] | None], provider: Optional[str], to_config: Callable[[T], Node] = ..., **to_config_kw: Any) -> T | ZenStore

        The interface to an initialized store. Can be used to store a config or to
        :ref:`customize the default values <self-partial>` of the store.

        Parameters
        ----------
        obj : Optional[T]
            The object to be stored. This is a **positional-only** argument.

            If `obj` is not specified, then the provided arguments are used to create a
            mirrored store instance with updated default arguments.

        name : NodeName | Callable[[T], NodeName]
            The entry's name, or a callable that will be called as
            `(obj) -> entry-name`. The default is `lambda obj: obj.__name__`.
            Store entries are keyed off of `(group, name)`.

        group : Optional[GroupName | Callable[[T], GroupName]]
            The entry's group's name, or a callable that will be called as
            `(obj) -> entry-group`. The default is `None`. Subgroups can be
            specified using / within the group name.

            Store entries are keyed off of `(group, name)`.

        to_config : Callable[[T], Node] = default_to_config
            Called on `obj` to produce the entry's "node" (the config).
            Refer to  `hydra_zen.wrapper.default_to_config` for the default
            behavior. Specify `lambda x: x` to have `obj` be stored directly
            as the entry's node.

            By default the call to `to_config` is deferred until the entry
            is actually accessed by the store.

        package : Optional[str | Callable[[Any], str]]
            The entry's package. Default is `None`.

        provider : Optional[str]
            An optional provider name for the entry.

        **to_config_kw : Any
            Additional arguments that will be passed to `to_config`.

        Returns
        -------
        T | ZenStore
            If `obj` was specified, it is returned unchanged. Otherwise a new instance
            of `ZenStore` is return, which mirrors the internal state of this store and
            has updated default arguments.
        """
        if __target is None:
            if self._warn_node_kwarg and "node" in kw:
                warnings.warn(
                    "hydra-zen's store API does not use the `node` keyword. To store a "
                    "config, specify it as a positional argument: `store(<config>)`."
                    "\n\nIf the use of `node` was intentional, you can suppress this "
                    "warning by using a store that is initialized via `ZenStore"
                    "(warn_node_kwarg=False)."
                )

            _s = type(self)(
                self.name,
                deferred_to_config=self._deferred_to_config,
                deferred_hydra_store=self._deferred_store,
                overwrite_ok=self._overwrite_ok,
                warn_node_kwarg=self._warn_node_kwarg,
            )
            _s._defaults = self._defaults.copy()

            # Important: mirror internal state *by reference* to ensure `_s` and
            # `self` remain in sync
            _s._internal_repo = self._internal_repo
            _s._queue = self._queue

            new_defaults: _StoreCallSig = {k: kw[k] for k in _DEFAULT_KEYS if k in kw}  # type: ignore

            new_defaults["__kw"] = {
                **_s._defaults["__kw"],
                **{k: kw[k] for k in set(kw) - _DEFAULT_KEYS},
            }
            _s._defaults.update(new_defaults)
            return _s
        else:
            to_config = kw.get("to_config", self._defaults["to_config"])
            name = kw.get("name", self._defaults["name"])
            group = kw.get("group", self._defaults["group"])
            package = kw.get("package", self._defaults["package"])
            provider = kw.get("provider", self._defaults["provider"])

            if (
                isinstance(__target, HydraConf)
                or isinstance(__target, type)
                and issubclass(__target, HydraConf)
            ):
                # User is re-configuring Hydra's config; we provide "smart" defaults
                # for the entry's name, group, and package
                if "name" not in kw and "group" not in kw:  # pragma: no branch
                    # only apply when neither name nor group are specified
                    name = "config"
                    group = "hydra"
                    if "provider" not in kw:  # pragma: no branch
                        provider = "hydra_zen"

            _name: NodeName = name(__target) if callable(name) else name
            if not isinstance(_name, str):
                raise TypeError(f"`name` must be a string, got {_name}")
            del name

            _group: GroupName = group(__target) if callable(group) else group
            if _group is not None and not isinstance(_group, str):
                raise TypeError(f"`group` must be a string or None, got {_group}")
            del group

            _pkg = package(__target) if callable(package) else package
            if _pkg is not None and not isinstance(_pkg, str):
                raise TypeError(f"`package` must be a string or None, got {_pkg}")
            del package

            merged_kw = {
                **self._defaults["__kw"],
                **{k: kw[k] for k in set(kw) - _DEFAULT_KEYS},
            }

            if self._deferred_to_config:
                node = _Deferred(to_config, __target, merged_kw)
            else:
                node = to_config(__target, **merged_kw)

            entry = StoreEntry(
                name=_name,
                group=_group,
                package=_pkg,
                provider=provider,
                node=node,
            )

            if not self._overwrite_ok and (_group, _name) in self._internal_repo:
                raise ValueError(
                    f"(name={entry['name']} group={entry['group']}): "
                    f"Store entry already exists. Use a store initialized "
                    f"with `ZenStore(overwrite_ok=True)` to overwrite config store "
                    f"entries."
                )
            self._internal_repo[_group, _name] = entry
            self._queue.append(entry)

            if not self._deferred_store:
                self.add_to_hydra_store()
            return __target

    @property
    def groups(self) -> Sequence[GroupName]:
        """Returns a sorted list of the groups registered with this store"""
        set_: Set[GroupName] = set(group for group, _ in self._internal_repo)
        if None in set_:
            set_.remove(None)
            no_none = cast(Set[str], set_)
            return [None] + sorted(no_none)
        else:
            no_none = cast(Set[str], set_)
            return sorted(no_none)

    def has_enqueued(self) -> bool:
        """`True` if this store has entries that have not yet been added to
        Hydra's config store.

        Returns
        -------
        bool

        Examples
        --------
        >>> from hydra_zen import ZenStore
        >>> store = ZenStore(deferred_hydra_store=True)
        >>> store.has_enqueued()
        False

        >>> store({"a": 1}, name)
        >>> store.has_enqueued()
        True

        >>> store.add_to_hydra_store()
        >>> store.has_enqueued()
        False
        """
        return bool(self._queue)

    def __bool__(self) -> bool:
        """`True` if entries have been added to this store, regardless of whether or
        not they have been added to Hydra's config store"""
        return bool(self._internal_repo)

    @overload
    def __getitem__(self, key: Tuple[GroupName, NodeName]) -> Node:
        ...

    @overload
    def __getitem__(self, key: GroupName) -> Dict[Tuple[GroupName, NodeName], Node]:
        ...

    def __getitem__(self, key: Union[GroupName, Tuple[GroupName, NodeName]]) -> Node:
        """Access a entry's config node by specifying `(group, name)`. Or, access a
        mapping of `(group, name) -> node` for all nodes in a specified group,
        including nodes within subgroups.

        See Also
        --------
        ZenStore.get_entry

        Examples
        --------
        >>> from hydra_zen import store
        >>> store(dict(x=1), name="a", group="fruit")
        >>> store(dict(x=2), name="b", group="fruit/apple")
        >>> store(dict(x=3), name="c", group="fruit/apple")
        >>> store(dict(x=4), name="d", group="fruit/orange")
        >>> store(dict(x=5), name="e", group="veggie")

        Accessing an individual entry's config node.

        >>> store["fruit/apple", "b"]
        {'x': 2}

        Accessing all config nodes under the "fruit/apple" group

        >>> store["fruit/apple"]
        {('fruit/apple', 'b'): {'x': 2}, ('fruit/apple', 'c'): {'x': 3}}

        Accessing all config nodes under the "fruit" group

        >>> store["fruit"]
        {('fruit', 'a'): {'x': 1},
         ('fruit/apple', 'b'): {'x': 2},
         ('fruit/apple', 'c'): {'x': 3},
         ('fruit/orange', 'd'): {'x': 4}}
        """
        # store[group] ->
        #  {(group, name): node1, (group, name2): node2, (group/subgroup, name3): node3}
        #
        # store[group, name] -> node
        if isinstance(key, str) or key is None:
            key_not_none = key is not None
            key_w_ender = key + "/" if key is not None else "<ZEN_NEVER>"
            return {
                (group, name): _resolve_node(entry, copy=False)["node"]
                for (group, name), entry in self._internal_repo.items()
                if group == key
                or (
                    key_not_none and group is not None and group.startswith(key_w_ender)
                )
            }
        return _resolve_node(self._internal_repo[key], copy=False)["node"]

    def get_entry(self, group: GroupName, name: NodeName) -> StoreEntry:
        """Access a store entry, which is a mapping that specifies the entry's
        name, group, package, provider, and node.

        Notes
        -----
        Mutating the returned mapping will not affect the store's internal entry.
        Mutating a node in the returned entry may have unintended consequences and
        is not advised.

        Examples
        --------
        >>> from hydra_zen import store, ZenStore
        >>> store(dict(x=1), name="a", group="fruit")
        >>> store.get_entry("fruit", "a")
        {'name': 'a',
         'group': 'fruit',
         'package': None,
         'provider': None,
         'node': {'x': 1}}
        """
        return _resolve_node(self._internal_repo[(group, name)], copy=True)

    def __contains__(self, key: Union[GroupName, Tuple[GroupName, NodeName]]) -> bool:
        """Checks if group or (group, node-name) exists in zen-store."""
        if key is None:
            return any(k[0] is None for k in self._internal_repo)  # pragma: no branch
        elif isinstance(key, str):
            key_w_end: str = key + "/"
            return any(
                key == group or group.startswith(key_w_end)
                for group, _ in self._internal_repo
                if group is not None
            )
        return key in self._internal_repo

    def __iter__(self) -> Generator[StoreEntry, None, None]:
        """Yields all entries in this store.

        Notes
        -----
        Mutating the returned mappings will not affect the store's internal entries.
        Mutating a node in an entry may have unintended consequences and is not advised.

        Examples
        --------
        >>> from hydra_zen import store
        >>> store(dict(x=1), name="a", group="fruit")
        >>> store(dict(x=2), name="b", group="fruit/orange")
        >>> store(dict(x=3), name="c", group="veggie")

        >>> list(store)
        [{'name': 'a',
          'group': 'fruit',
          'package': None,
          'provider': None,
          'node': {'x': 1}},
         {'name': 'b',
          'group': 'fruit/orange',
          'package': None,
          'provider': None,
          'node': {'x': 2}},
         {'name': 'c',
          'group': 'veggie',
          'package': None,
          'provider': None,
          'node': {'x': 3}}]
        """
        yield from (_resolve_node(v, copy=True) for v in self._internal_repo.values())

    def add_to_hydra_store(self, overwrite_ok: Optional[bool] = None) -> None:
        """Adds all of this store's enqueued entries to Hydra's global config store.

        This method need not be called for a store initialized as
        `ZenStore(deferred_hydra_store=False)`.

        Parameters
        ----------
        overwrite_ok : Optional[bool]
            If `False`, this method raises `ValueError` if an entry in Hydra's config
            store will be overwritten. Defaults to the value of `overwrite_ok`
            specified when initializing this store.

        Examples
        --------
        >>> from hydra_zen import ZenStore
        >>> store1 = ZenStore()
        >>> store2 = ZenStore()

        >>> store1({'a': 1}, name="x")
        >>> store1.add_to_hydra_store()
        >>> store2({'a': 2}, name="x")
        >>> store2.add_to_hydra_store()
        ValueError: (name=x group=None): Hydra config store entry already exists. Specify `overwrite_ok=True` to enable replacing config store entries

        >>> store2.add_to_hydra_store(overwrite_ok=True)  # successfully overwrites entry

        """
        _store = ConfigStore.instance().store
        while self._queue:
            entry = _resolve_node(self._queue.popleft(), copy=False)
            if (
                (
                    overwrite_ok is False
                    or (overwrite_ok is None and not self._overwrite_ok)
                )
                and self._exists_in_hydra_store(
                    name=entry["name"], group=entry["group"]
                )
                # It is okay if we are overwriting Hydra's default store
                and not (
                    (entry["name"], entry["group"]) == ("config", "hydra")
                    and ConfigStore.instance().repo["hydra"]["config.yaml"].provider
                    == "hydra"
                )
            ):
                raise ValueError(
                    f"(name={entry['name']} group={entry['group']}): "
                    f"Hydra config store entry already exists. Specify "
                    f"`overwrite_ok=True` to enable replacing config store entries"
                )
            _store(**entry)

    def _exists_in_hydra_store(
        self,
        *,
        name: NodeName,
        group: GroupName,
        hydra_store: ConfigStore = ConfigStore().instance(),
    ) -> bool:
        repo = hydra_store.repo

        if group is not None:
            for group_name in group.split("/"):
                repo = repo.get(group_name)
                if repo is None:
                    return False
        return name + ".yaml" in repo


store: ZenStore = ZenStore(
    name="zen_store",
    deferred_to_config=True,
    deferred_hydra_store=True,
)
