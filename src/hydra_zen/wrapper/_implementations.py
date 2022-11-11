# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIR
# pyright: strict

from collections import defaultdict, deque
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
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing_extensions import Final, Literal, ParamSpec, TypeAlias, TypedDict, TypeGuard

from hydra_zen import instantiate, just, make_custom_builds_fn
from hydra_zen.errors import HydraZenValidationError
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
        Generates a Hydra-CLI for the wrapped function. Equivalent to `hydra.main(zen(func), [...])()`

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

        kw = dict(config_name=config_name)

        if config_path is not _UNSPECIFIED_:
            kw["config_path"] = config_path

        if SUPPORTS_VERSION_BASE and version_base is not _UNSPECIFIED_:
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
    ConfigLike is DataClass | dict[str, Any] | DictConfig

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


    **Passing Through The Config**

    Some task functions require complete access to the full config to gain access to
    sub-configs. One can specify the field named `zen_config` in their task function's
    signature to signal `zen` that it should pass the full config to that parameter .

    >>> @zen
    ... def f(x: int, zen_cfg):
    ...     return x, zen_cfg
    >>> f(dict(x=1, y="${x}"))
    (1, {'x': 1, 'y': 1})

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
        Callable[..., Any], DataClass_, List[Any], Dict[Any, Any], ListConfig
    ],
    **kw: Any,
) -> Union[DataClass_, Type[DataClass_], ListConfig, DictConfig]:
    if is_dataclass(target):
        if isinstance(target, type):
            if get_obj_path(target).startswith("types."):
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
        return just(target)
    elif isinstance(target, (DictConfig, ListConfig)):
        return target
    else:
        t = cast(Callable[..., Any], target)
        return fbuilds(t, **kw)


def get_name(target: Any) -> str:
    name = getattr(target, "__name__", None)
    if not isinstance(name, str):
        raise TypeError(
            f"Cannot infer config store entry name for {target}. Please manually "
            f"specify `store({target}, name=<some name>, [...])`"
        )
    return name


class _Defaults(TypedDict):
    name: Union[NodeName, Callable[[Any], NodeName]]
    group: Union[GroupName, Callable[[Any], GroupName]]
    package: Optional[Union[str, Callable[[Any], str]]]
    provider: Optional[str]
    __kw: Dict[str, Any]
    to_config: Callable[[Any], Any]


# TODO: make frozen dict
defaults: Final = _Defaults(
    name=get_name,
    group=None,
    package=None,
    provider=None,
    to_config=default_to_config,
    __kw={},
)

_DEFAULT_KEYS: Final[FrozenSet[str]] = frozenset(_Defaults.__required_keys__ - {"__kw"})  # type: ignore


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


def _resolve_node(entry: StoreEntry) -> StoreEntry:
    """Given an entry, updates the entry so that its node is not deferred, and returns
    the entry. This function is a passthrough for an entry whose node is not deferred"""
    item = entry["node"]
    if isinstance(item, _Deferred):
        entry["node"] = item()
    return entry


class ZenStore:
    __slots__ = (
        "name",
        "_internal_repo",
        "_defaults",
        "_queue",
        "_deferred_to_config",
        "_deferred_store",
        "_overwrite_ok",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        deferred_to_config: bool = True,
        deferred_hydra_store: bool = True,
        overwrite_ok: bool = False,
    ) -> None:
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
        self._internal_repo: Dict[Tuple[Optional[str], str], StoreEntry] = {}
        self._queue: Deque[StoreEntry] = deque([])
        self._defaults = defaults.copy()
        self._deferred_to_config = deferred_to_config
        self._deferred_store = deferred_hydra_store
        self._overwrite_ok = overwrite_ok

    def __repr__(self) -> str:
        groups_contents: DefaultDict[Optional[str], List[str]] = defaultdict(list)
        for grp, name in self._internal_repo:
            groups_contents[grp].append(name)
        return f"({self.name}){dict(groups_contents)}"

    # TODO: support *to_config_pos_args
    @overload
    def __call__(
        self,
        __target: F,
        *,
        name: Union[NodeName, Callable[[Any], NodeName]] = ...,
        group: Union[GroupName, Callable[[Any], GroupName]] = ...,
        package: Optional[Union[str, Callable[[Any], str]]] = ...,
        provider: Optional[str] = ...,
        to_config: Callable[[F], Node] = default_to_config,
        **to_config_kw: Any,
    ) -> F:  # pragma: no cover
        ...

    # TODO: ZenStore -> Self when mypy adds support for Self
    #       https://github.com/python/mypy/pull/11666
    @overload
    def __call__(
        self,
        __target: Literal[None] = None,
        *,
        name: Union[NodeName, Callable[[Any], NodeName]] = ...,
        group: Union[GroupName, Callable[[Any], GroupName]] = ...,
        package: Optional[Union[str, Callable[[Any], str]]] = ...,
        provider: Optional[str] = ...,
        to_config: Callable[[Any], Node] = ...,
        **to_config_kw: Any,
    ) -> "ZenStore":  # pragma: no cover
        ...

    def __call__(self, __target: Optional[F] = None, **kw: Any) -> Union[F, "ZenStore"]:
        if __target is None:
            _s = type(self)(
                self.name,
                deferred_to_config=self._deferred_to_config,
                deferred_hydra_store=self._deferred_store,
                overwrite_ok=self._overwrite_ok,
            )
            _s._defaults = self._defaults.copy()

            # Important: mirror internal state *by reference* to ensure `_s` and
            # `self` remain in sync
            _s._internal_repo = self._internal_repo
            _s._queue = self._queue

            new_defaults: _Defaults = {k: kw[k] for k in _DEFAULT_KEYS if k in kw}  # type: ignore

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

            _name: NodeName = name(__target) if callable(name) else name
            if not isinstance(_name, str):  # type: ignore
                raise TypeError(f"`name` must be a string, got {_name}")
            del name

            _group: GroupName = group(__target) if callable(group) else group
            if _group is not None and not isinstance(_group, str):  # type: ignore
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
                    f"Hydra config store entry already exists. Specify "
                    f"`overwrite_ok=True` to enable replacing config store entries"
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
    def __getitem__(self, key: Tuple[GroupName, NodeName]) -> Node:  # pragma: no cover
        ...

    @overload
    def __getitem__(
        self, key: GroupName
    ) -> Dict[Tuple[GroupName, NodeName], Node]:  # pragma: no cover
        ...

    def __getitem__(self, key: Union[GroupName, Tuple[GroupName, NodeName]]) -> Node:
        # store[group] ->
        #  {(group, name): node1, (group, name2): node2, (group/subgroup, name3): node3}
        #
        # store[group, name] -> node
        if isinstance(key, str) or key is None:
            key_not_none = key is not None
            key_w_ender = key + "/" if key is not None else "<NEVER>"
            return {
                (group, name): _resolve_node(entry)["node"]
                for (group, name), entry in self._internal_repo.items()
                if group == key
                or (
                    key_not_none and group is not None and group.startswith(key_w_ender)
                )
            }
        return self.get_entry(*key)["node"]

    def get_entry(self, group: GroupName, name: NodeName) -> StoreEntry:
        return _resolve_node(self._internal_repo[(group, name)])

    def __contains__(self, key: Union[GroupName, Tuple[GroupName, NodeName]]) -> bool:
        """Checks if group or (group, node-name) exists in zen-store"""
        if key is None:
            return any(k[0] is None for k in self._internal_repo)

        if isinstance(key, str):
            key_w_end: str = key + "/"
            return any(
                key == group or group.startswith(key_w_end)
                for group, _ in self._internal_repo
                if group is not None
            )
        return key in self._internal_repo

    def __iter__(self) -> Generator[StoreEntry, None, None]:
        yield from (_resolve_node(v) for v in self._internal_repo.values())

    def add_to_hydra_store(self, overwrite_ok: Optional[bool] = None) -> None:

        while self._queue:
            entry = _resolve_node(self._queue.popleft())
            if (
                overwrite_ok is False
                or (overwrite_ok is None and not self._overwrite_ok)
            ) and self._exists_in_hydra_store(name=entry["name"], group=entry["group"]):
                raise ValueError(
                    f"(name={entry['name']} group={entry['group']}): "
                    f"Hydra config store entry already exists. Specify "
                    f"`overwrite_ok=True` to enable replacing config store entries"
                )
            ConfigStore.instance().store(**entry)

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
    name="store", deferred_to_config=True, deferred_hydra_store=True
)
