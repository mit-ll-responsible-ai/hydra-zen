# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
import inspect
import pathlib
import sys
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import (  # use this for runtime checks
    MISSING,
    Field as _Field,
    InitVar,
    dataclass,
    field,
    fields,
    is_dataclass,
    make_dataclass,
)
from datetime import timedelta
from enum import Enum
from functools import partial
from itertools import chain
from pathlib import Path, PosixPath, WindowsPath
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from omegaconf import DictConfig, ListConfig, _utils as _omegaconf_utils
from typing_extensions import (
    Concatenate,
    Literal,
    ParamSpec,
    ParamSpecArgs,
    ParamSpecKwargs,
    Protocol,
    Self,
    TypeAlias,
    Unpack,
    _AnnotatedAlias,
    dataclass_transform,
)

from hydra_zen._compatibility import (
    HYDRA_SUPPORTED_PRIMITIVE_TYPES,
    HYDRA_SUPPORTED_PRIMITIVES,
    ZEN_SUPPORTED_PRIMITIVES,
)
from hydra_zen.errors import (
    HydraZenDeprecationWarning,
    HydraZenUnsupportedPrimitiveError,
    HydraZenValidationError,
)
from hydra_zen.funcs import as_default_dict, get_obj
from hydra_zen.structured_configs import _utils
from hydra_zen.structured_configs._type_guards import safe_getattr
from hydra_zen.typing import (
    Builds,
    DataclassOptions,
    PartialBuilds,
    SupportedPrimitive,
    ZenConvert,
    ZenWrappers,
)
from hydra_zen.typing._implementations import (
    AllConvert,
    AnyBuilds,
    Builds,
    BuildsWithSig,
    DataClass,
    DataClass_,
    DataclassOptions,
    DefaultsList,
    Field,
    HasTarget,
    HasTargetInst,
    HydraSupportedType,
    InstOrType,
    Just as JustT,
    Partial,
    ZenConvert,
    _HydraPrimitive,
    _SupportedViaBuilds,
)

from ._globals import (
    CONVERT_FIELD_NAME,
    DEFAULTS_LIST_FIELD_NAME,
    HYDRA_FIELD_NAMES,
    JUST_FIELD_NAME,
    META_FIELD_NAME,
    PARTIAL_FIELD_NAME,
    POS_ARG_FIELD_NAME,
    RECURSIVE_FIELD_NAME,
    TARGET_FIELD_NAME,
    ZEN_PARTIAL_FIELD_NAME,
    ZEN_PROCESSING_LOCATION,
    ZEN_TARGET_FIELD_NAME,
    ZEN_WRAPPERS_FIELD_NAME,
)
from ._type_guards import (
    is_builds,
    is_generic_type,
    is_just,
    is_old_partial_builds,
    safe_getattr,
    uses_zen_processing,
)
from ._utils import merge_settings

T = TypeVar("T")
_T = TypeVar("_T")
P = ParamSpec("P")
R = TypeVar("R")

TD = TypeVar("TD", bound=DataClass_)
TC = TypeVar("TC", bound=Callable[..., Any])
TP = TypeVar("TP", bound=_HydraPrimitive)
TB = TypeVar("TB", bound=Union[_SupportedViaBuilds, frozenset[Any]])

Importable = TypeVar("Importable", bound=Callable[..., Any])
Field_Entry: TypeAlias = tuple[str, type, Field[Any]]

_JUST_CONVERT_SETTINGS = AllConvert(dataclass=True, flat_target=False)

# default zen_convert settings for `builds` and `hydrated_dataclass`
_BUILDS_CONVERT_SETTINGS = AllConvert(dataclass=True, flat_target=True)


# stores type -> value-conversion-fn
# for types with specialized support from hydra-zen
class _ConversionFn(Protocol):
    def __call__(self, __x: Any, CBuildsFn: "Type[BuildsFn[Any]]") -> Any: ...


ZEN_VALUE_CONVERSION: dict[type, _ConversionFn] = {}

# signature param-types
_POSITIONAL_ONLY: Final = inspect.Parameter.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD: Final = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL: Final = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY: Final = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD: Final = inspect.Parameter.VAR_KEYWORD


NoneType = type(None)
_supported_types = HYDRA_SUPPORTED_PRIMITIVE_TYPES | {
    list,
    dict,
    tuple,
    List,
    Tuple,
    Dict,
}


_builtin_function_or_method_type = type(len)
# fmt: off
_lru_cache_type = type(functools.lru_cache(maxsize=128)(lambda: None))  # pragma: no branch
# fmt: on

_BUILTIN_TYPES: Final = (_builtin_function_or_method_type, _lru_cache_type)

del _lru_cache_type
del _builtin_function_or_method_type


# In python 3.13 the definitions for pathlib.Path, et al. were moved to the
# pathlib._locals, changing the tag for yaml serialization. Thus we monkey-patch
# omegaconf's yaml loader to handle these new strings.
_original_yaml_loader = _omegaconf_utils.get_yaml_loader


def _patched_yaml_loader(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    loader = _original_yaml_loader(*args, **kwargs)
    loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib._local.Path",
        lambda loader, node: pathlib.Path(*loader.construct_sequence(node)),
    )
    loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib._local.PosixPath",
        lambda loader, node: pathlib.PosixPath(*loader.construct_sequence(node)),
    )
    loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib._local.WindowsPath",
        lambda loader, node: pathlib.WindowsPath(*loader.construct_sequence(node)),
    )
    return loader


_omegaconf_utils.get_yaml_loader = _patched_yaml_loader


def _retain_type_info(type_: type, value: Any, hydra_recursive: Optional[bool]):
    # OmegaConf's type-checking occurs before instantiation occurs.
    # This means that, e.g., passing `Builds[int]` to a field `x: int`
    # will fail Hydra's type-checking upon instantiation, even though
    # the recursive instantiation will appropriately produce `int` for
    # that field. This will not be addressed by hydra/omegaconf:
    #    https://github.com/facebookresearch/hydra/issues/1759
    # Thus we will auto-broaden the annotation when we see that a field
    # is set with a structured config as a default value - assuming that
    # the field isn't annotated with a structured config type.

    # Each condition is included separately to ensure that our tests
    # cover all scenarios
    if hydra_recursive is False:
        return True
    elif not is_builds(value):
        if _utils.is_interpolated_string(value):
            # an interpolated field may resolve to a structured conf, which may
            # instantiate to a value of the specified type
            return False
        return True
    elif is_builds(type_):
        return True
    return False


@dataclass_transform()
def hydrated_dataclass(
    target: Callable[..., Any],
    *pos_args: SupportedPrimitive,
    zen_partial: Optional[bool] = None,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    zen_convert: Optional[ZenConvert] = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = True,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> Callable[[type[_T]], type[_T]]:
    """A decorator that uses `builds` to create a dataclass with the appropriate
    Hydra-specific fields for specifying a targeted config [1]_.

    This provides similar functionality to `builds`, but enables a user to define
    a config explicitly using the :func:`dataclasses.dataclass` syntax, which can
    enable enhanced static analysis of the resulting config.

    Parameters
    ----------
    hydra_target : T (Callable)
        The target-object to be configured. This is a required, positional-only argument.

    *pos_args : SupportedPrimitive
        Positional arguments passed as ``hydra_target(*pos_args, ...)`` upon instantiation.

        Arguments specified positionally are not included in the dataclass' signature
        and are stored as a tuple bound to in the ``_args_`` field.

    zen_partial : Optional[bool]
        If ``True``, then the resulting config will instantiate as
        ``functools.partial(hydra_target, *pos_args, **kwargs_for_target)`` rather than
        ``hydra_target(*pos_args, **kwargs_for_target)``. Thus this enables the
        partial-configuration of objects.

        Specifying ``zen_partial=True`` and ``populate_full_signature=True`` together
        will populate the config's signature only with parameters that: are explicitly
        specified by the user, or that have default values specified in the target's
        signature. I.e. it is presumed that un-specified parameters that have no
        default values are to be excluded from the config.

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
        E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

            f3(f2(f1(hydra_target)))(*args, **kwargs)

        Wrappers can also be specified as interpolated strings [2]_ or targeted
        configs.

    zen_meta : Optional[Mapping[str, SupportedPrimitive]]
        Specifies field-names and corresponding values that will be included in the
        resulting config, but that will *not* be used to builds ``<hydra_target>``
        via instantiation. These are called "meta" fields.

    populate_full_signature : bool, optional (default=False)
        If ``True``, then the resulting config's signature and fields will be populated
        according to the signature of ``hydra_target``; values also specified in
        ``**kwargs_for_target`` take precedent.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    zen_convert : Optional[ZenConvert]
        A dictionary that modifies hydra-zen's value and type conversion behavior.
        Consists of the following optional key-value pairs (:ref:`zen-convert`):

        - `dataclass` : `bool` (default=True):
            If `True` any dataclass type/instance without a
            `_target_` field is automatically converted to a targeted config
            that will instantiate to that type/instance. Otherwise the dataclass
            type/instance will be passed through as-is.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this config [3]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

        - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
        - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
          ``list``, respectively. Structured configs and their fields are passed without conversion.
        - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without a trace of OmegaConf containers.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    hydra_convert : Optional[Literal["none", "partial", "all", "object"]], optional (default="none")
        Determines how Hydra treats the non-primitive, omegaconf-specific objects
        during instantiateion [3]_.

        - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
        - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
          ``list``, respectively. Structured configs and their fields are passed without conversion.
        - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without
          a trace of OmegaConf containers.
        - ``"object"``: Passed objects are converted to dict and list. Structured Configs are converted to instances of the backing dataclass / attr class.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    init : bool, optional (default=True)
        If true (the default), a __init__() method will be generated. If the class
        already defines __init__(), this parameter is ignored.

    repr : bool, optional (default=True)
        If true (the default), a `__repr__()` method will be generated. The generated
        repr string will have the class name and the name and repr of each field, in
        the order they are defined in the class. Fields that are marked as being
        excluded from the repr are not included. For example:
        `InventoryItem(name='widget', unit_price=3.0, quantity_on_hand=10)`.

    eq : bool, optional (default=True)
        If true (the default), an __eq__() method will be generated. This method
        compares the class as if it were a tuple of its fields, in order. Both
        instances in the comparison must be of the identical type.

    order : bool, optional (default=False)
        If true (the default is `False`), `__lt__()`, `__le__()`, `__gt__()`, and
        `__ge__()` methods will be generated. These compare the class as if it were a
        tuple of its fields, in order. Both instances in the comparison must be of the
        identical type. If order is true and eq is false, a ValueError is raised.

        If the class already defines any of `__lt__()`, `__le__()`, `__gt__()`, or
        `__ge__()`, then `TypeError` is raised.

    unsafe_hash : bool, optional (default=False)
        If `False` (the default), a `__hash__()` method is generated according to how
        `eq` and `frozen` are set.

        If `eq` and `frozen` are both true, by default `dataclass()` will generate a
        `__hash__()` method for you. If `eq` is true and `frozen` is false, `__hash__()
        ` will be set to `None`, marking it unhashable. If `eq` is false, `__hash__()`
        will be left untouched meaning the `__hash__()` method of the superclass will
        be used (if the superclass is object, this means it will fall back to id-based
        hashing).

    frozen : bool, optional (default=False)
        If true (the default is `False`), assigning to fields will generate an
        exception. This emulates read-only frozen instances.

    match_args : bool, optional (default=True)
        (*New in version 3.10*) If true (the default is `True`), the `__match_args__`
        tuple will be created from the list of parameters to the generated `__init__()`
        method (even if `__init__()` is not generated, see above). If false, or if
        `__match_args__` is already defined in the class, then `__match_args__` will
        not be generated.

    kw_only : bool, optional (default=False)
        (*New in version 3.10*) If true (the default value is `False`), then all fields
        will be marked as keyword-only.

    slots : bool, optional (default=False)
        (*New in version 3.10*) If true (the default is `False`), `__slots__` attribute
        will be generated and new class will be returned instead of the original one.
        If `__slots__` is already defined in the class, then `TypeError` is raised.

    weakref_slot : bool, optional (default=False)
        (*New in version 3.11*) If true (the default is `False`), add a slot named
        “__weakref__”, which is required to make an instance weakref-able. It is an
        error to specify `weakref_slot=True` without also specifying `slots=True`.


    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError
        The provided configured value cannot be serialized by Hydra, nor does hydra-zen
        provide specialized support for it. See :ref:`valid-types` for more details.

    Notes
    -----
    Unlike `builds`, `hydrated_dataclass` enables config fields to be set explicitly
    with custom type annotations. Additionally, the resulting config' attributes
    can be analyzed by static tooling, which can help to warn about errors prior
    to running one's code.

    For details of the annotation `SupportedPrimitive`, see :ref:`valid-types`.

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/structured_config/intro/
    .. [2] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    .. [3] https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [4] https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen import hydrated_dataclass, instantiate

    Here, we specify a config that is designed to "build" a dictionary
    upon instantiation

    >>> @hydrated_dataclass(target=dict, frozen=True)
    ... class DictConf:
    ...     x: int = 2
    ...     y: str = 'hello'

    >>> instantiate(DictConf(x=10))  # override default `x`
    {'x': 10, 'y': 'hello'}

    >>> d = DictConf()
    >>> # Static type checker marks the following as
    >>> # an error because `d` is frozen.
    >>> d.x = 3  # type: ignore
    FrozenInstanceError: cannot assign to field 'x'

    For more detailed examples, refer to `builds`.
    """

    def wrapper(decorated_obj: Any) -> Any:
        if not isinstance(decorated_obj, type):
            raise NotImplementedError(
                "Class instances are not supported by `hydrated_dataclass`."
            )

        # TODO: We should mutate `decorated_obj` directly like @dataclass does.
        #       Presently, we create an intermediate dataclass that we inherit
        #       from, which gets the job done for the most part but there are
        #       practical differences. E.g. you cannot delete an attribute that
        #       was declared in the definition of `decorated_obj`.
        dc_options = _utils.parse_dataclass_options(
            {
                "init": init,
                "repr": repr,
                "eq": eq,
                "order": order,
                "unsafe_hash": unsafe_hash,
                "frozen": frozen,
                "match_args": match_args,
                "kw_only": kw_only,
                "slots": slots,
                "weakref_slot": weakref_slot,
            },
            include_module=False,
        )
        decorated_obj = dataclass(**dc_options)(decorated_obj)  # type: ignore

        if populate_full_signature:
            # we need to ensure that the fields specified via the class definition
            # take precedence over the fields that will be auto-populated by builds
            kwargs = {
                f.name: f.default if f.default is not MISSING else f.default_factory()  # type: ignore
                for f in fields(decorated_obj)
                if not (f.default is MISSING and f.default_factory is MISSING)
                and f.name not in HYDRA_FIELD_NAMES
                and not f.name.startswith("_zen_")
            }
        else:
            kwargs: dict[str, Any] = {}

        out = DefaultBuilds.builds(
            target,
            *pos_args,
            **kwargs,
            populate_full_signature=populate_full_signature,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            zen_wrappers=zen_wrappers,
            zen_partial=zen_partial,
            zen_meta=zen_meta,
            builds_bases=(decorated_obj,),
            zen_dataclass={
                "cls_name": decorated_obj.__name__,
                "module": decorated_obj.__module__,
                "init": init,
                "repr": repr,
                "eq": eq,
                "order": order,
                "unsafe_hash": unsafe_hash,
                "frozen": frozen,
                "match_args": match_args,
                "kw_only": kw_only,
                "slots": slots,
                "weakref_slot": weakref_slot,
            },
            zen_convert=zen_convert,
        )
        if decorated_obj.__doc__ is not None:  # pragma: no cover
            out.__doc__ = decorated_obj.__doc__
        return out

    return wrapper


@dataclass(unsafe_hash=True)
class Just:
    """Just[T] is a config that returns T when instantiated."""

    path: str
    _target_: str = field(default="hydra_zen.funcs.get_obj", init=False, repr=False)


def _is_ufunc(value: Any) -> bool:
    # checks without importing numpy
    if (numpy := sys.modules.get("numpy")) is None:  # pragma: no cover
        # we do actually cover this branch some runs of our CI,
        # but our coverage job installs numpy
        return False
    return isinstance(value, numpy.ufunc)


def _is_jax_ufunc(value: Any) -> bool:  # pragma: no cover
    # checks without importing numpy
    if (jnp := sys.modules.get("jax.numpy")) is None:  # pragma: no cover
        return False
    return isinstance(value, jnp.ufunc)


def _is_numpy_array_func_dispatcher(value: Any) -> bool:
    if (numpy := sys.modules.get("numpy")) is None:  # pragma: no cover
        return False
    return isinstance(value, type(numpy.sum))


def _check_instance(*target_types: str, value: "Any", module: str):  # pragma: no cover
    """Checks if value is an instance of any of the target types imported
    from the specified module.

    Returns `False` if module/target type doesn't exists (e.g. not installed).
    This is useful for gracefully handling specialized logic for optional dependencies.
    """
    if (mod := sys.modules.get(module)) is None:
        return False

    types = []
    for attr_name in target_types:
        type_ = getattr(mod, attr_name, None)
        if type_ is not None:
            types.append(type_)

    if not types:
        return False

    try:
        return isinstance(value, tuple(types))
    except TypeError:
        # handle singleton checking
        return any(value is t for t in types)


_is_jax_compiled_func = functools.partial(
    _check_instance, "CompiledFunction", "PjitFunction", module="jaxlib.xla_extension"
)

_is_jax_compiled_func2 = functools.partial(
    _check_instance, "CompiledFunction", "PjitFunction", module="jaxlib._jax"
)
_is_jax_unspecified = functools.partial(
    _check_instance, "UnspecifiedValue", module="jax._src.interpreters.pxla"
)

_is_torch_optim_required = functools.partial(
    _check_instance, "required", module="torch.optim.optimizer"
)

_is_pydantic_BaseModel = functools.partial(
    _check_instance, "BaseModel", module="pydantic"
)


def _check_for_dynamically_defined_dataclass_type(target_path: str, value: Any) -> None:
    if target_path.startswith("types."):
        raise HydraZenUnsupportedPrimitiveError(
            f"Configuring {value}: Cannot auto-config an instance of a "
            f"dynamically-generated dataclass type (e.g. one created from "
            f"`hydra_zen.make_config` or `dataclasses.make_dataclass`). "
            f"Consider disabling auto-config support for dataclasses here."
        )


class NOTHING:
    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("`NOTHING` cannot be instantiated")


@dataclass(unsafe_hash=True)
class ZenField:
    """
    ZenField(hint=Any, default=<class 'NOTHING'>, name=<class 'NOTHING'>)

    Specifies a field's name and/or type-annotation and/or default value.
    Designed to specify fields in `make_config`.

    See the Examples section of the docstring for `make_config` for examples of using
    `ZenField`.

    Parameters
    ----------
    hint : type, optional (default=Any)
    default : Any, optional
    name : str, optional

    Notes
    -----
    ``default`` will be returned as an instance of :class:`dataclasses.Field`.
    Mutable values (e.g. lists or dictionaries) passed to ``default`` will automatically
    be "packaged" in a default-factory function [1]_.

    A type passed to ``hint`` will automatically be "broadened" such that the resulting
    type is compatible with Hydra's set of supported type annotations [2]_.

    References
    ----------
    .. [1] https://docs.python.org/3/library/dataclasses.html#default-factory-functions
    .. [2] https://hydra.cc/docs/next/tutorials/structured_config/intro/#structured-configs-supports

    See Also
    --------
    make_config: create a config with customized field names, default values, and annotations.
    """

    hint: Any = Any
    default: Union[Any, Field[Any]] = _utils.field(default=NOTHING)
    name: Union[str, type[NOTHING]] = NOTHING
    zen_convert: InitVar[Optional[ZenConvert]] = None
    _builds_fn: "Union[BuildsFn[Any], Type[BuildsFn[Any]]]" = _utils.field(default_factory=lambda: DefaultBuilds)  # type: ignore

    def __post_init__(
        self,
        zen_convert: Optional[ZenConvert],
    ) -> None:
        if not isinstance(self.name, str):
            if self.name is not NOTHING:
                raise TypeError(f"`ZenField.name` expects a string, got: {self.name}")
        convert_settings = _utils.merge_settings(zen_convert, _BUILDS_CONVERT_SETTINGS)
        del zen_convert

        self.hint = self._builds_fn._sanitized_type(self.hint)

        if self.default is not NOTHING:
            self.default = self._builds_fn._sanitized_field(
                self.default,
                convert_dataclass=convert_settings["dataclass"],
            )


_MAKE_CONFIG_SETTINGS = AllConvert(dataclass=False, flat_target=False)


class BuildsFn(Generic[T]):
    """A class that can be modified to customize the behavior of `builds`, `just`, `kwargs_of`, and `make_config`.

    These functions are exposed as class methods of `BuildsFn`.

    - To customize type-refinement support, override `_sanitized_type`.
    - To customize auto-config support, override `_make_hydra_compatible`.
    - To customize the ability to resolve import paths, override `_get_obj_path`.

    Notes
    -----
    Adding type-checking support for a custom type:

    To parameterize `BuildsFn` with, e.g., support for the custom types `MyTypeA` and `MyTypeB`, use::

       from typing import Union
       from hydra_zen.typing import CustomConfigType

       class MyTypeA: ...
       class MyTypeB: ...

       class MyBuilds(BuildsFn[CustomConfigType[Union[MyTypeA, MyTypeB]]]):
           ...

    Examples
    --------
    Suppose you wrote the following type::

       class Quaternion:
           def __init__(
               self,
               real: float = 0.0,
               i: float = 0.0,
               j: float = 0.0,
               k: float = 0.0,
           ):
               self._data = (real, i, j, k)

           def __repr__(self):
               return "Q" + repr(self._data)

    and you want hydra-zen's config-creation functions to be able to automatically
    know how to make configs from instances of this type. You do so by creating your
    own subclass of :class:`~hydra_zen.BuildsFn`::

       from typing import Any
       from hydra_zen import BuildsFn
       from hydra_zen.typing import CustomConfigType, HydraSupportedType

       class CustomBuilds(BuildsFn[CustomConfigType[Quaternion]]):
           @classmethod
           def _make_hydra_compatible(cls, value: Any, **k) -> HydraSupportedType:
               if isinstance(value, Quaternion):
                   real, i, j, k = value._data
                   return cls.builds(Quaternion, real=real, i=i, j=j, k=k)
               return super()._make_hydra_compatible(value, **k)

    Now you use the config-creation functions that are provided by `CustomBuilds` instead of those provided by `hydra_zen`::

       builds = CustomBuilds.builds
       just = CustomBuilds.just
       kwargs_of = CustomBuilds.kwargs_of
       make_config = CustomBuilds.make_config

    E.g.

    >>> from hydra_zen import to_yaml
    >>> Config = just([Quaternion(1.0), Quaternion(0.0, -12.0)])
    >>> print(to_yaml(Config))
    - _target_: __main__.Quaternion
      real: 1.0
      i: 0.0
      j: 0.0
      k: 0.0
    - _target_: __main__.Quaternion
      real: 0.0
      i: -12.0
      j: 0.0
      k: 0.0
    """

    __slots__ = ()

    _default_dataclass_options_for_kwargs_of: Optional[DataclassOptions] = None
    """Specifies the default options for `cls.kwargs_of(..., zen_dataclass)"""

    @classmethod
    def _sanitized_type(
        cls,
        type_: Any,
        *,
        primitive_only: bool = False,
        wrap_optional: bool = False,
        nested: bool = False,
    ) -> Any:  # is really type[Any]
        """Broadens a type annotation until it is compatible with Hydra.

        Override this to change how `builds` refines the type annotations
        of the configs that it produces.

        Parameters
        ----------
        type_ : Any
            The type being sanitized.

        primitive_only: bool, optional (default=False)
            If true, only `bool` | `None` | `int` | `float` | str`
            is permitted.

        wrap_optional: bool, optional (default=False)
            `True` indicates that the resulting type should be
            wrapped in `Optional`.

        nested: bool, optional (default=False)
            `True` indicates that this function is processing a type within
            a container type.

        Examples
        --------
        >>> sanitized_type(int)
        <class 'int'>

        >>> sanitized_type(frozenset)  # not supported by hydra
        typing.Any

        >>> sanitized_type(int, wrap_optional=True)
        typing.Union[int, NoneType]

        >>> sanitized_type(List[int])
        List[int]

        >>> sanitized_type(List[int], primitive_only=True)
        Any

        >>> sanitized_type(Dict[str, frozenset])
        Dict[str, Any]
        """
        if hasattr(type_, "__supertype__"):
            # is NewType
            return cls._sanitized_type(
                type_.__supertype__,
                primitive_only=primitive_only,
                wrap_optional=wrap_optional,
                nested=nested,
            )

        # Warning: mutating `type_` will mutate the signature being inspected
        # Even calling deepcopy(`type_`) silently fails to prevent this.
        origin = get_origin(type_)

        if origin is not None:
            # Support for Annotated[x, y]
            # Python 3.9+
            # # type_: Annotated[x, y]; origin -> Annotated; args -> (x, y)
            if origin is Annotated:  # pragma: no cover
                return cls._sanitized_type(
                    get_args(type_)[0],
                    primitive_only=primitive_only,
                    wrap_optional=wrap_optional,
                    nested=nested,
                )

            # Python 3.7-3.8
            # type_: Annotated[x, y]; origin -> x
            if isinstance(type_, _AnnotatedAlias):  # pragma: no cover
                return cls._sanitized_type(
                    origin,
                    primitive_only=primitive_only,
                    wrap_optional=wrap_optional,
                    nested=nested,
                )

            if primitive_only:  # pragma: no cover
                return Any

            args = get_args(type_)
            if origin is Union:
                # Hydra only supports Optional[<type>] unions
                if len(args) != 2 or NoneType not in args:
                    # isn't Optional[<type>]
                    return Any

                args = cast(tuple[type, type], args)

                optional_type, none_type = args
                if none_type is not NoneType:
                    optional_type = none_type

                optional_type = cls._sanitized_type(optional_type)

                if optional_type is Any:  # Union[Any, T] is just Any
                    return Any

                return cast(type, Union[optional_type, NoneType])

            if origin is list or origin is List:
                if args:
                    return list[
                        cls._sanitized_type(args[0], primitive_only=False, nested=True)
                    ]
                return list

            if origin is dict or origin is Dict:
                if args:
                    KeyType = cls._sanitized_type(
                        args[0], primitive_only=True, nested=True
                    )
                    ValueType = cls._sanitized_type(
                        args[1], primitive_only=False, nested=True
                    )
                    return dict[KeyType, ValueType]
                return dict

            if (origin is tuple or origin is Tuple) and not nested:
                # hydra silently supports tuples of homogeneous types
                # It has some weird behavior. It treats `Tuple[t1, t2, ...]` as `List[t1]`
                # It isn't clear that we want to perpetrate this on our end..
                # So we deal with inhomogeneous types as e.g. `Tuple[str, int]` -> `Tuple[Any, Any]`.
                #
                # Otherwise we preserve the annotation as accurately as possible
                if not args:
                    return tuple

                args = cast(tuple[type, ...], args)
                unique_args = set(args)

                if any(get_origin(tp) is Unpack for tp in unique_args):
                    # E.g. Tuple[*Ts]
                    return tuple[Any, ...]

                has_ellipses = Ellipsis in unique_args

                # E.g. Tuple[int, int, int] or Tuple[int, ...]
                _unique_type = (
                    cls._sanitized_type(args[0], primitive_only=False, nested=True)
                    if len(unique_args) == 1 or (len(unique_args) == 2 and has_ellipses)
                    else Any
                )

                if has_ellipses:
                    return tuple[_unique_type, ...]
                else:
                    return tuple[(_unique_type,) * len(args)]

            return Any

        if isinstance(type_, type) and issubclass(type_, Path):
            type_ = Path

        if isinstance(type_, (ParamSpecArgs, ParamSpecKwargs)):  # pragma: no cover
            # Python 3.7 - 3.9
            # these aren't hashable -- can't check for membership in set
            return Any

        if isinstance(type_, InitVar):
            return cls._sanitized_type(
                type_.type,
                primitive_only=primitive_only,
                wrap_optional=wrap_optional,
                nested=nested,
            )
        if (
            type_ is Any
            or type_ in _supported_types
            or is_dataclass(type_)
            or (isinstance(type_, type) and issubclass(type_, Enum))
        ):

            if wrap_optional and type_ is not Any:  # pragma: no cover
                # normally get_type_hints automatically resolves Optional[...]
                # when None is set as the default, but this has been flaky
                # for some pytorch-lightning classes. So we just do it ourselves...
                # It might be worth removing this later since none of our standard tests
                # cover it.
                type_ = Optional[type_]
            return type_

        return Any

    @classmethod
    def _get_obj_path(cls, target: Any) -> str:
        """Used to get the `_target_` value for the resulting config.

        Override this to control how `builds` determines the _target_ field
        in the configs that it produces."""

        name = _utils.safe_name(target, repr_allowed=False)

        if name == _utils.UNKNOWN_NAME:
            if is_generic_type(target):  # pragma: no cover
                return cls._get_obj_path(target.__origin__)

            raise AttributeError(f"{target} does not have a `__name__` attribute")

        module = getattr(target, "__module__", None)
        qualname: Union[str, None] = getattr(target, "__qualname__", None)

        if (qualname is not None and "<" in qualname) or module is None:
            # NumPy's ufuncs do not have an inspectable `__module__` attribute, so we
            # check to see if the object lives in NumPy's top-level namespace.
            #
            # or..
            #
            # Qualname produced a name from a local namespace.
            # E.g. jax.numpy.add.__qualname__ is '_maybe_bool_binop.<locals>.fn'
            # Thus we defer to the name of the object and look for it in the
            # top-level namespace of the known suspects
            #
            # or...
            #
            # module is None, which is apparently a thing..:
            # __module__ is None for both numpy.random.rand and random.random
            #

            # don't use qualname for obfuscated paths
            for new_module in _utils.COMMON_MODULES_WITH_OBFUSCATED_IMPORTS:
                if getattr(sys.modules.get(new_module), name, None) is target:
                    module = new_module
                    break
            else:
                raise ModuleNotFoundError(f"{name} is not importable")

        if not _utils.is_classmethod(target):
            if (
                (inspect.isfunction(target) or isinstance(target, type))
                and isinstance(qualname, str)
                and "." in qualname
                and all(x.isidentifier() for x in qualname.split("."))
            ):
                # This looks like it is a staticmethod or a class defined within
                # a class namespace. E.g. qualname: SomeClass.func or
                # SomeClass.NestedClass
                return f"{module}.{qualname}"
            return f"{module}.{name}"
        else:
            # __qualname__ reflects name of class that originally defines classmethod.
            # Does not point to child in case of inheritance.
            #
            # obj.__self__ -> parent object
            # obj.__name__ -> name of classmethod
            return f"{cls._get_obj_path(target.__self__)}.{target.__name__}"

    @classmethod
    def _just(cls, obj: Any) -> Just:
        return Just(path=cls._get_obj_path(obj))

    @classmethod
    def _mutable_value(cls, x: _T, *, zen_convert: Optional[ZenConvert] = None) -> _T:
        """Used to set a mutable object as a default value for a field
        in a dataclass.

        This is an alias for ``field(default_factory=lambda: type(x)(x))``

        Note that ``type(x)(...)`` serves to make a copy

        Examples
        --------
        >>> from hydra_zen import mutable_value
        >>> from dataclasses import dataclass

        See https://docs.python.org/3/library/dataclasses.html#mutable-default-values

        >>> @dataclass  # doctest: +SKIP
        ... class HasMutableDefault:
        ...     a_list: list  = [1, 2, 3]  # error: mutable default

        Using `mutable_value` to specify the default list:

        >>> @dataclass
        ... class HasMutableDefault:
        ...     a_list: list  = mutable_value([1, 2, 3])  # ok

        >>> x = HasMutableDefault()
        >>> x.a_list.append(-1)  # does not append to `HasMutableDefault.a_list`
        >>> x
        HasMutableDefault(a_list=[1, 2, 3, -1])
        >>> HasMutableDefault()
        HasMutableDefault(a_list=[1, 2, 3])"""
        cast = type(x)  # ensure that we return a copy of the default value
        settings = _utils.merge_settings(zen_convert, _BUILDS_CONVERT_SETTINGS)
        del zen_convert

        if cast in {list, tuple, dict}:
            x = cls._sanitize_collection(x, convert_dataclass=settings["dataclass"])
            return field(default_factory=lambda: cast(x))  # type: ignore
        return field(default_factory=lambda: x)

    @classmethod
    def _make_hydra_compatible(
        cls,
        value: object,
        *,
        allow_zen_conversion: bool = True,
        error_prefix: str = "",
        field_name: str = "",
        structured_conf_permitted: bool = True,
        convert_dataclass: bool,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
    ) -> HydraSupportedType:
        """Converts `value` to Hydra-supported type or to a config whose fields are recursively supported by `_make_hydra_compatible`. Otherwise
        raises `HydraZenUnsupportedPrimitiveError`.

        Override this method to add support for adding auto-config support to custom
        types.

        Notes
        -----
        Hydra supports the following types:
            `bool`, `None`, `int`, `float`, `str`, `ByteString`, `pathlib.Path`,
            `dataclasses.MISSING`. As well as lists, tuples, dicts, and omegaconf
            containers containing the above.
        """
        from hydra_zen.wrapper import Zen

        # Common primitives supported by Hydra.
        # We check exhaustively for all Hydra-supported primitives below but seek to
        # speedup checks for common types here.
        if value is None or type(value) in {str, int, bool, float}:
            return cast(Union[None, str, int, float, bool], value)

        # non-str collection
        if hasattr(value, "__iter__"):
            value = cls._sanitize_collection(
                value,
                convert_dataclass=convert_dataclass,
                hydra_convert=hydra_convert,
                hydra_recursive=hydra_recursive,
            )

        if zen_dataclass is None:
            zen_dataclass = {}

        # non-targeted dataclass instance
        if (
            structured_conf_permitted
            and convert_dataclass
            and not is_builds(value)
            and (is_dataclass(value) and not isinstance(value, type))
        ):
            # Auto-config dataclass instance
            # TODO: handle position-only arguments
            _val_fields = fields(value)
            if set(inspect.signature(type(value)).parameters) != {
                f.name for f in _val_fields if f.init
            }:
                raise HydraZenUnsupportedPrimitiveError(
                    f"Configuring {value}: Cannot auto-config a dataclass instance whose "
                    f"type has an Init-only field. Consider using "
                    f"`builds({type(value).__name__}, ...)` instead."
                )

            converted_fields = {}
            for _field in _val_fields:
                if _field.init and hasattr(value, _field.name):
                    _val = safe_getattr(value, _field.name)
                    converted_fields[_field.name] = cls._make_hydra_compatible(
                        _val,
                        allow_zen_conversion=allow_zen_conversion,
                        field_name=_field.name,
                        convert_dataclass=convert_dataclass,
                    )

            out = cls.builds(
                type(value),
                **converted_fields,
                hydra_recursive=hydra_recursive,
                hydra_convert=hydra_convert,
                zen_dataclass=zen_dataclass,
            )
            _check_for_dynamically_defined_dataclass_type(
                safe_getattr(out, TARGET_FIELD_NAME), value
            )
            return out

        # importable callable (function, type, or method)
        if (
            structured_conf_permitted
            and callable(value)
            and (
                inspect.isfunction(value)
                or (
                    (
                        not is_dataclass(value)
                        or (convert_dataclass and not is_builds(value))
                    )
                    and (inspect.isclass(value) or is_generic_type(value))
                )
                or inspect.ismethod(value)
                or isinstance(value, _BUILTIN_TYPES)
                or _is_ufunc(value)
                or _is_numpy_array_func_dispatcher(value=value)
                or _is_jax_compiled_func(value=value)
                or _is_jax_compiled_func2(value=value)
                or _is_jax_ufunc(value=value)
            )
        ):
            # `value` is importable callable -- create config that will import
            # `value` upon instantiation
            out = cls._just(value)
            if convert_dataclass and is_dataclass(value):
                _check_for_dynamically_defined_dataclass_type(
                    safe_getattr(out, JUST_FIELD_NAME), value
                )
            return out

        if isinstance(value, Zen):
            pre_call = [cls.just(f) for f in value._pre_call_iterable if f]
            if not pre_call:  # pragma: no cover
                pre_call = None
            elif len(pre_call) == 1:  # pragma: no cover
                pre_call = pre_call[0]

            return cls.builds(
                type(value),
                value.func,  # type: ignore
                exclude=list(value._exclude),  # type: ignore
                pre_call=pre_call,  # type: ignore
                unpack_kwargs=value._unpack_kwargs,  # type: ignore
                resolve_pre_call=value._resolve,  # type: ignore
                run_in_context=value._run_in_context,  # type: ignore
                instantiation_wrapper=value._instantiation_wrapper,  # type: ignore
                populate_full_signature=True,
            )
        resolved_value = value
        type_of_value = type(resolved_value)

        # hydra-zen supported primitives from stdlib
        #
        # Note: we don't use isinstance because we don't permit subclasses of supported
        # primitives
        if allow_zen_conversion and type_of_value in ZEN_SUPPORTED_PRIMITIVES:
            type_ = type(resolved_value)
            conversion_fn = ZEN_VALUE_CONVERSION[type_]

            resolved_value = conversion_fn(resolved_value, CBuildsFn=cls)
            type_of_value = type(resolved_value)

        if type_of_value in HYDRA_SUPPORTED_PRIMITIVES or (
            structured_conf_permitted
            and (
                is_dataclass(resolved_value)
                or isinstance(resolved_value, (Enum, ListConfig, DictConfig))
            )
        ):
            return resolved_value  # type: ignore

        # pydantic objects
        pydantic = sys.modules.get("pydantic")

        if pydantic is not None:  # pragma: no cover
            if _check_instance("FieldInfo", module="pydantic.fields", value=value):
                _val = (
                    value.default_factory()  # type: ignore
                    if value.default_factory is not None  # type: ignore
                    else value.default  # type: ignore
                )

                if _check_instance(
                    "UndefinedType", module="pydantic.fields", value=_val
                ):
                    return MISSING

                return cls._make_hydra_compatible(
                    _val,
                    allow_zen_conversion=allow_zen_conversion,
                    error_prefix=error_prefix,
                    field_name=field_name,
                    structured_conf_permitted=structured_conf_permitted,
                    convert_dataclass=convert_dataclass,
                    hydra_convert=hydra_convert,
                    hydra_recursive=hydra_recursive,
                )
            if _is_pydantic_BaseModel(value=value):
                return cls.builds(type(value), **value.__dict__)

        if isinstance(value, str) or _check_instance(
            "AnyUrl", module="pydantic", value=value
        ):
            # Supports pydantic.AnyURL
            _v = str(value)
            if type(_v) is str:  # pragma: no branch
                return _v
            else:  # pragma: no cover
                del _v

        # support for torch/jax MISSING proxies
        if _is_torch_optim_required(value=value) or _is_jax_unspecified(
            value=value
        ):  # pragma: no cover
            return MISSING

        # `value` could no be converted to Hydra-compatible representation.
        # Raise error
        if field_name:
            field_name = f", for field `{field_name}`,"

        err_msg = (
            error_prefix
            + f" The configured value {value}{field_name} is not supported by Hydra -- "
            f"serializing or instantiating this config would ultimately result in an error."
        )

        if structured_conf_permitted:
            err_msg += (
                f"\n\nConsider using `hydra_zen.builds({type(value)}, ...)` create "
                "a config for this particular value."
            )

        raise HydraZenUnsupportedPrimitiveError(err_msg)

    @classmethod
    def _sanitize_collection(
        cls,
        x: _T,
        *,
        convert_dataclass: bool,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    ) -> _T:
        """Pass contents of lists, tuples, or dicts through sanitized_default_values"""
        type_x = type(x)
        if type_x in {list, tuple}:
            return type_x(
                cls._make_hydra_compatible(
                    _x,
                    convert_dataclass=convert_dataclass,
                    hydra_convert=hydra_convert,
                    hydra_recursive=hydra_recursive,
                )
                for _x in x  # type: ignore
            )
        elif type_x is dict:
            return {
                # Hydra doesn't permit structured configs for keys, thus we only
                # support its basic primitives here.
                cls._make_hydra_compatible(
                    k,
                    allow_zen_conversion=False,
                    structured_conf_permitted=False,
                    error_prefix="Configuring dictionary key:",
                    convert_dataclass=False,
                ): cls._make_hydra_compatible(
                    v,
                    convert_dataclass=convert_dataclass,
                    hydra_convert=hydra_convert,
                    hydra_recursive=hydra_recursive,
                )
                for k, v in x.items()  # type: ignore
            }
        else:
            # pass-through
            return x

    @classmethod
    def _sanitized_field(
        cls,
        value: Any,
        init: bool = True,
        allow_zen_conversion: bool = True,
        *,
        error_prefix: str = "",
        field_name: str = "",
        convert_dataclass: bool,
    ) -> Field[Any]:
        value = cls._make_hydra_compatible(
            value,
            allow_zen_conversion=allow_zen_conversion,
            error_prefix=error_prefix,
            field_name=field_name,
            convert_dataclass=convert_dataclass,
        )

        type_value = type(value)
        if (
            type_value in _utils.KNOWN_MUTABLE_TYPES
            and type_value in HYDRA_SUPPORTED_PRIMITIVES
        ) or (
            is_dataclass(value)
            and not isinstance(value, type)
            and value.__hash__ is None
        ):
            return cast(
                Field[Any],
                mutable_value(value, zen_convert={"dataclass": convert_dataclass}),
            )
        return _utils.field(default=value, init=init)

    @classmethod
    def _get_sig_obj(cls, target: Any) -> Any:
        """Return the signature object for `target`.

        `inspect.signature` has inconsistent/buggy behaviors across
        versions, so we implement our own."""
        if not inspect.isclass(target):
            return target

        # This implements the same method prioritization as
        # `inspect.signature` for Python >= 3.9.1
        if "__new__" in target.__dict__:
            return target.__new__
        if "__init__" in target.__dict__:
            return target.__init__

        if len(target.__mro__) > 2:
            for parent in target.__mro__[1:-1]:
                if "__new__" in parent.__dict__:
                    return target.__new__
                elif "__init__" in parent.__dict__:
                    return target.__init__
        return getattr(target, "__init__", target)

    # partial=False, pop-sig=True; no *args, **kwargs, nor builds_bases
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: type[BuildsWithSig[type[R], P]],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: Callable[P, R],
        *,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: Literal[True],
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[()] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
    ) -> type[BuildsWithSig[type[R], P]]: ...

    # partial=False, pop-sig=bool
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[Builds[Importable]]: ...

    # partial=False, pop-sig=bool
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[False, None] = ...,
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[Builds[Importable]]: ...

    # partial=True, pop-sig=bool
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Literal[True],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    # partial=True, pop-sig=bool
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Literal[True],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> type[PartialBuilds[Importable]]: ...

    # partial=bool, pop-sig=False
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: type[AnyBuilds[Importable]],
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[type[Builds[Importable]], type[PartialBuilds[Importable]]]: ...

    # partial=bool, pop-sig=False
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: Importable,
        *pos_args: T,
        zen_partial: Optional[bool] = ...,
        populate_full_signature: Literal[False] = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[type[Builds[Importable]], type[PartialBuilds[Importable]]]: ...

    # partial=bool, pop-sig=bool
    @overload
    @classmethod
    def builds(
        cls: type[Self],
        __hydra_target: Union[Callable[P, R], type[Builds[Importable]], Importable],
        *pos_args: T,
        zen_partial: Optional[bool],
        populate_full_signature: bool = ...,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        hydra_defaults: Optional[DefaultsList] = ...,
        dataclass_name: Optional[str] = ...,
        builds_bases: tuple[type[DataClass_], ...] = ...,
        zen_dataclass: Optional[DataclassOptions] = None,
        frozen: bool = ...,
        zen_convert: Optional[ZenConvert] = ...,
        **kwargs_for_target: T,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]: ...

    @classmethod
    def builds(
        cls: type[Self],
        *pos_args: Union[
            Importable,
            Callable[P, R],
            type[AnyBuilds[Importable]],
            type[BuildsWithSig[type[R], P]],
            Any,
        ],
        zen_partial: Optional[bool] = None,
        zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
        zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
        populate_full_signature: bool = False,
        zen_convert: Optional[ZenConvert] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        builds_bases: Union[tuple[type[DataClass_], ...], tuple[()]] = (),
        zen_dataclass: Optional[DataclassOptions] = None,
        **kwargs_for_target: Any,
    ) -> Union[
        type[Builds[Importable]],
        type[PartialBuilds[Importable]],
        type[BuildsWithSig[type[R], P]],
    ]:
        """builds(hydra_target, /, *pos_args, zen_partial=None, zen_wrappers=(), zen_meta=None, populate_full_signature=False, zen_exclude=(), hydra_recursive=None, hydra_convert=None, hydra_defaults=None, builds_bases=(),
        zen_dataclass=None, **kwargs_for_target)

        `builds(target, *args, **kw)` returns a Hydra-compatible config that, when
        instantiated, returns `target(*args, **kw)`.

        I.e., `instantiate(builds(target, *args, **kw)) == target(*args, **kw)`

        Consult the Notes section for more details, and the Examples section to see
        the various features of `builds` in action.

        Parameters
        ----------
        hydra_target : T (Callable)
            The target object to be configured. This is a required, **positional-only**
            argument.

        *pos_args : SupportedPrimitive
            Positional arguments passed as ``<hydra_target>(*pos_args, ...)`` upon
            instantiation.

            Arguments specified positionally are not included in the config's signature
            and are stored as a tuple bound to in the ``_args_`` field.

        **kwargs_for_target : SupportedPrimitive
            The keyword arguments passed as ``<hydra_target>(..., **kwargs_for_target)``
            upon instantiation.

            The arguments specified here determine the signature of the resulting
            config, unless ``populate_full_signature=True`` is specified (see below).

            Named parameters of the forms that have the prefixes ``hydra_``, ``zen_`` or
            ``_zen_`` are reserved to ensure future-compatibility, and thus cannot be
            specified by the user.

        zen_partial : Optional[bool]
            If ``True``, then the resulting config will instantiate as
            ``functools.partial(<hydra_target>, *pos_args, **kwargs_for_target)``. Thus
            this enables the partial-configuration of objects.

            Specifying ``zen_partial=True`` and ``populate_full_signature=True`` together
            will populate the config's signature only with parameters that: are explicitly
            specified by the user, or that have default values specified in the target's
            signature. I.e. it is presumed that un-specified parameters that have no
            default values are to be excluded from the config.

        zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
            One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
            E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

                f3(f2(f1(<hydra_target>)))(*args, **kwargs)

            Wrappers can also be specified as interpolated strings [2]_ or targeted
            configs.

        zen_meta : Optional[Mapping[str, SupportedPrimitive]]
            Specifies field-names and corresponding values that will be included in the
            resulting config, but that will *not* be used to instantiate
            ``<hydra_target>``. These are called "meta" fields.

        populate_full_signature : bool, optional (default=False)
            If ``True``, then the resulting config's signature and fields will be
            populated according to the signature of ``<hydra_target>``; values also
            specified in ``**kwargs_for_target`` take precedent.

            This option is not available for objects with inaccessible signatures, such
            as NumPy's various ufuncs.

        zen_exclude : Collection[str | int] | Callable[[str], bool], optional (default=[])
            Specifies parameter names and/or indices, or a function for checking names,
            to exclude those parameters from the config-creation process.

        Note that inherited fields cannot be excluded.
        zen_convert : Optional[ZenConvert]
            A dictionary that modifies hydra-zen's value and type conversion behavior.
            Consists of the following optional key-value pairs (:ref:`zen-convert`):

            - `dataclass` : `bool` (default=True):
                If `True` any dataclass type/instance without a
                `_target_` field is automatically converted to a targeted config
                that will instantiate to that type/instance. Otherwise the dataclass
                type/instance will be passed through as-is.

            - `flat_target`: `bool` (default=True)
                If `True` (default), `builds(builds(f))` is equivalent to `builds(f)`. I.e. the second `builds` call will use the `_target_` field of its input, if it exists.

        builds_bases : Tuple[Type[DataClass], ...]
            Specifies a tuple of parent classes that the resulting config inherits from.

        hydra_recursive : Optional[bool], optional (default=True)
            If ``True``, then Hydra will recursively instantiate all other
            hydra-config objects nested within this config [3]_.

            If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

        hydra_convert : Optional[Literal["none", "partial", "all", "object"]], optional (default="none")
            Determines how Hydra handles the non-primitive, omegaconf-specific objects passed to
            ``<hydra_target>`` [4]_.

            - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
            - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
            ``list``, respectively. Structured configs and their fields are passed without conversion.
            - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without
            a trace of OmegaConf containers.
            - ``"object"``: Passed objects are converted to dict and list. Structured Configs are converted to instances of the backing dataclass / attr class.

            If ``None``, the ``_convert_`` attribute is not set on the resulting config.

        hydra_defaults : None | list[str | dict[str, str | list[str] | None ]], optional (default = None)
            A list in an input config that instructs Hydra how to build the output config
            [6]_ [7]_. Each input config can have a Defaults List as a top level element. The
            Defaults List itself is not a part of output config.

        zen_dataclass : Optional[DataclassOptions]
            A dictionary that can specify any option that is supported by
            :py:func:`dataclasses.make_dataclass` other than `fields`.
            The default value for `unsafe_hash` is `True`.

            `target` can be specified as a string to override the `_target_` field
            set on the dataclass type returned by `builds`.

            The `module` field can be specified to enable pickle
            compatibility. See `hydra_zen.typing.DataclassOptions` for details.

        frozen : bool, optional (default=False)
            .. deprecated:: 0.9.0
                `frozen` will be removed in hydra-zen 0.10.0. It is replaced by
                `zen_dataclass={'frozen': <bool>}`.

            If ``True``, the resulting config will create frozen (i.e. immutable) instances.
            I.e. setting/deleting an attribute of an instance will raise
            :py:class:`dataclasses.FrozenInstanceError` at runtime.

        dataclass_name : Optional[str]
            .. deprecated:: 0.9.0
                `dataclass_name` will be removed in hydra-zen 0.10.0. It is replaced by
                `zen_dataclass={'cls_name': <str>}`.

            If specified, determines the name of the returned class object.

        Returns
        -------
        Config : Type[Builds[Type[T]]] | Type[PartialBuilds[Type[T]]]
            A dynamically-generated structured config (i.e. a dataclass type) that
            describes how to build ``hydra_target``.

        Raises
        ------
        hydra_zen.errors.HydraZenUnsupportedPrimitiveError
            The provided configured value cannot be serialized by Hydra, nor does
            hydra-zen provide specialized support for it. See :ref:`valid-types` for
            more details.

        Notes
        -----
        The following pseudo code conveys the core functionality of `builds`:

        .. code-block:: python

            from dataclasses import make_dataclass

            def builds(self,target, populate_full_signature=False, **kw):
                # Dynamically defines a Hydra-compatible dataclass type.
                # Akin to doing:
                #
                # @dataclass
                # class Builds_thing:
                #     _target_: str = get_import_path(target)
                #     # etc.

                _target_ = get_import_path(target)

                if populate_full_signature:
                    sig = get_signature(target)
                    kw = {**sig, **kw}  # merge w/ preference for kw

                type_annots = [get_hints(target)[k] for k in kw]

                fields = [("_target_", str, _target_)]
                fields += [
                    (
                        field_name,
                        hydra_compat_type_annot(hint),
                        hydra_compat_val(v),
                    )
                    for hint, (field_name, v) in zip(type_annots, kw.items())
                ]

                Config = make_dataclass(f"Builds_{target}", fields)
                return Config

        The resulting "config" is a dynamically-generated dataclass type [5]_ with
        Hydra-specific attributes attached to it [1]_. It possesses a `_target_`
        attribute that indicates the import path to the configured target as a string.

        Using any of the ``zen_xx`` features will result in a config that depends
        explicitly on hydra-zen. I.e. hydra-zen must be installed in order to
        instantiate the resulting config, including its yaml version.

        For details of the annotation `SupportedPrimitive`, see :ref:`valid-types`.

        Type annotations are inferred from the target's signature and are only
        retained if they are compatible with Hydra's limited set of supported
        annotations; otherwise an annotation is automatically 'broadened' until
        it is made compatible with Hydra.

        `builds` provides runtime validation of user-specified arguments against
        the target's signature. E.g. specifying mis-named arguments or too many
        arguments will cause `builds` to raise.

        References
        ----------
        .. [1] https://hydra.cc/docs/tutorials/structured_config/intro/
        .. [2] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
        .. [3] https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
        .. [4] https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies
        .. [5] https://docs.python.org/3/library/dataclasses.html
        .. [6] https://hydra.cc/docs/tutorials/structured_config/defaults/
        .. [7] https://hydra.cc/docs/advanced/defaults_list/

        See Also
        --------
        instantiate: Instantiates a configuration created by `builds`, returning the instantiated target.
        make_custom_builds_fn: Returns a new `builds` function with customized default values.
        make_config: Creates a general config with customized field names, default values, and annotations.
        get_target: Returns the target-object from a targeted structured config.
        just: Produces a config that, when instantiated by Hydra, "just" returns the un-instantiated target-object.
        to_yaml: Serialize a config as a yaml-formatted string.

        Examples
        --------
        These examples describe:

        - Basic usage
        - Creating a partial config
        - Auto-populating parameters
        - Composing configs via inheritance
        - Runtime validation performed by builds
        - Using meta-fields
        - Using zen-wrappers
        - Creating a pickle-compatible config
        - Creating a frozen config
        - Support for partial'd targets

        A helpful utility for printing examples

        >>> from hydra_zen import builds, instantiate, to_yaml
        >>> def pyaml(x):
        ...     # for pretty printing configs
        ...     print(to_yaml(x))

        **Basic Usage**

        Lets create a basic config that describes how to 'build' a particular dictionary.

        >>> Conf = builds(dict, a=1, b='x')

        The resulting config is a dataclass with the following signature and attributes:

        >>> Conf  # signature: Conf(a: Any = 1, b: Any = 'x')
        <class 'types.Builds_dict'>

        >>> pyaml(Conf)
        _target_: builtins.dict
        a: 1
        b: x

        The `instantiate` function is used to enact this build – to create the dictionary.

        >>> instantiate(Conf)  # calls: `dict(a=1, b='x')`
        {'a': 1, 'b': 'x'}

        The default parameters that we provided can be overridden.

        >>> new_conf = Conf(a=10, b="hi")  # an instance of our dataclass
        >>> instantiate(new_conf)  # calls: `dict(a=10, b='hi')`
        {'a': 10, 'b': 'hi'}

        Positional arguments are supported.

        >>> Conf = builds(len, [1, 2, 3])
        >>> Conf._args_  # type: ignore
        [1, 2, 3]
        >>> instantiate(Conf)
        3

        **Creating a Partial Config**

        `builds` can be used to partially-configure a target. Let's
        create a config for the following function

        >>> def a_two_tuple(x: int, y: float): return x, y

        such that we only configure the parameter ``x``.

        >>> PartialConf = builds(a_two_tuple, x=1, zen_partial=True)  # configures only `x`
        >>> pyaml(PartialConf)
        _target_: __main__.a_two_tuple
        _partial_: true
        x: 1

        Instantiating this config will return ``functools.partial(a_two_tuple, x=1)``.

        >>> partial_func = instantiate(PartialConf)
        >>> partial_func
        functools.partial(<function a_two_tuple at 0x00000220A7820EE0>, x=1)

        And thus the remaining parameter can be provided post-instantiation.

        >>> partial_func(y=22.0)  # providing the remaining parameter
        (1, 22.0)

        **Auto-populating parameters**

        The configurable parameters of a target can be auto-populated in our config.
        Suppose we want to configure the following function.

        >>> def bar(x: bool, y: str = 'foo'): return x, y

        The following config will have a signature that matches ``bar``; the
        annotations and default values of the parameters of ``bar`` are explicitly
        incorporated into the config.

        >>> # signature: `Builds_bar(x: bool, y: str = 'foo')`
        >>> Conf = builds(bar, populate_full_signature=True)
        >>> pyaml(Conf)
        _target_: __main__.bar
        x: ???
        'y': foo

        `zen_exclude` can be used to name parameter to be excluded from the
        auto-population process:

        >>> Conf2 = builds(bar, populate_full_signature=True, zen_exclude=["y"])
        >>> pyaml(Conf2)
        _target_: __main__.bar
        x: ???

        to exclude parameters by index:

        >>> Conf2 = builds(bar, populate_full_signature=True, zen_exclude=[-1])
        >>> pyaml(Conf2)
        _target_: __main__.bar
        x: ???

        or to specify a pattern - via a function - for excluding parameters:

        >>> Conf3 = builds(bar, populate_full_signature=True,
        ...                zen_exclude=lambda name: name.startswith("x"))
        >>> pyaml(Conf3)
        _target_: __main__.bar
        'y': foo

        Annotations will be used by Hydra to provide limited runtime type-checking
        during instantiation. Here, we'll pass a float for ``x``, which expects a
        boolean value.

        >>> instantiate(Conf(x=10.0))  # type: ignore
        ValidationError: Value '10.0' is not a valid bool (type float)
            full_key: x
            object_type=Builds_func

        **Composing configs via inheritance**

        Because a config produced via `builds` is simply a class-object, we can
        compose configs via class inheritance.

        >>> ParentConf = builds(dict, a=1, b=2)
        >>> ChildConf = builds(dict, b=-2, c=-3, builds_bases=(ParentConf,))
        >>> instantiate(ChildConf)
        {'a': 1, 'b': -2, 'c': -3}
        >>> issubclass(ChildConf, ParentConf)  # type: ignore
        True

        .. _builds-validation:

        **Runtime validation performed by builds**

        Misspelled parameter names and other invalid configurations for the target’s
        signature will be caught by `builds` so that such errors are caught prior to
        instantiation.

        >>> def func(a_number: int): pass

        >>> builds(func, a_nmbr=2)  # misspelled parameter name
        TypeError: Building: func ..

        >>> builds(func, 1, 2)  # too many arguments
        TypeError: Building: func ..

        >>> BaseConf = builds(func, a_number=2)
        >>> builds(func, 1, builds_bases=(BaseConf,))  # too many args (via inheritance)
        TypeError: Building: func ..

        >>> # value type not supported by Hydra
        >>> builds(int, (i for i in range(10)))  # type: ignore
        hydra_zen.errors.HydraZenUnsupportedPrimitiveError: Building: int ..


        .. _meta-field:

        **Using meta-fields**

        Meta-fields are fields that are included in a config but are excluded by the
        instantiation process. Thus arbitrary metadata can be attached to a config.

        Let's create a config whose fields reference a meta-field via
        relative-interpolation [2]_.

        >>> Conf = builds(dict, a="${.s}", b="${.s}", zen_meta=dict(s=-10))
        >>> instantiate(Conf)
        {'a': -10, 'b': -10}
        >>> instantiate(Conf, s=2)
        {'a': 2, 'b': 2}

        .. _zen-wrapper:

        **Using zen-wrappers**

        Zen-wrappers enables us to make arbitrary changes to ``<hydra_target>``, its inputs,
        and/or its outputs during the instantiation process.

        Let's use a wrapper to add a unit-conversion step to a config. We'll modify a
        config that builds a function, which converts a temperature in Fahrenheit to
        Celsius, and add a wrapper to it so that it will convert from Fahrenheit to
        Kelvin instead.

        >>> def faren_to_celsius(temp_f):  # our target
        ...     return ((temp_f - 32) * 5) / 9

        >>> def change_celcius_to_kelvin(celc_func):  # our wrapper
        ...     def wraps(*args, **kwargs):
        ...         return 273.15 + celc_func(*args, **kwargs)
        ...     return wraps

        >>> AsCelcius = builds(faren_to_celsius)
        >>> AsKelvin = builds(faren_to_celsius, zen_wrappers=change_celcius_to_kelvin)
        >>> instantiate(AsCelcius, temp_f=32)
        0.0
        >>> instantiate(AsKelvin, temp_f=32)
        273.15

        **Creating a pickle-compatible config**

        The dynamically-generated classes created by `builds` can be made pickle-compatible
        by specifying the name of the symbol that it is assigned to and the module in which
        it was defined.

        .. code-block:: python

           # contents of mylib/foo.py
           from pickle import dumps, loads

           DictConf = builds(dict,
                               zen_dataclass={'module': 'mylib.foo',
                                               'cls_name': 'DictConf'})

           assert DictConf is loads(dumps(DictConf))


        **Creating a frozen config**

        Let's create a config object whose instances will by "frozen" (i.e., immutable).

        >>> RouterConfig = builds(dict, ip_address=None, zen_dataclass={'frozen': True})
        >>> my_router = RouterConfig(ip_address="192.168.56.1")  # an immutable instance

        Attempting to overwrite the attributes of ``my_router`` will raise.

        >>> my_router.ip_address = "148.109.37.2"
        FrozenInstanceError: cannot assign to field 'ip_address'

        **Support for partial'd targets**

        Specifying ``builds(functools.partial(<target>, ...), ...)`` is supported; `builds`
        will automatically "unpack" a partial'd object that is passed as its target.

        >>> import functools
        >>> partiald_dict = functools.partial(dict, a=1, b=2)
        >>> Conf = builds(partiald_dict)  # signature: (a = 1, b = 2)
        >>> instantiate(Conf)  # equivalent to calling: `partiald_dict()`
        {'a': 1, 'b': 2}
        >>> instantiate(Conf(a=-4))  # equivalent to calling: `partiald_dict(a=-4)`
        {'a': -4, 'b': 2}
        """

        zen_convert_settings = _utils.merge_settings(
            zen_convert, _BUILDS_CONVERT_SETTINGS
        )
        if zen_dataclass is None:
            zen_dataclass = {}

        # initial validation
        _utils.parse_dataclass_options(zen_dataclass)

        manual_target_path = zen_dataclass.pop("target", None)
        target_repr = zen_dataclass.pop("target_repr", True)

        if "frozen" in kwargs_for_target:
            warnings.warn(
                HydraZenDeprecationWarning(
                    "Specifying `builds(..., frozen=<...>)` is deprecated. Instead, "
                    "specify `builds(..., zen_dataclass={'frozen': <...>})"
                ),
                stacklevel=2,
            )
            zen_dataclass["frozen"] = kwargs_for_target.pop("frozen")

        if "dataclass_name" in kwargs_for_target:
            warnings.warn(
                HydraZenDeprecationWarning(
                    "Specifying `builds(..., dataclass_name=<...>)` is deprecated. "
                    "Instead specify `builds(..., zen_dataclass={'cls_name': <...>})"
                ),
                stacklevel=2,
            )
            zen_dataclass["cls_name"] = kwargs_for_target.pop("dataclass_name")
        if not builds_bases:
            builds_bases = zen_dataclass.get("bases", ())

        dataclass_options = _utils.parse_dataclass_options(zen_dataclass)
        dataclass_name = dataclass_options.pop("cls_name", None)
        module = dataclass_options.pop("module", None)

        del zen_convert

        if not pos_args and not kwargs_for_target:
            # `builds()`
            raise TypeError(
                "builds() missing 1 required positional argument: 'hydra_target'"
            )
        elif not pos_args:
            # `builds(hydra_target=int)`
            raise TypeError(
                "builds() missing 1 required positional-only argument: 'hydra_target'"
                "\nChange `builds(hydra_target=<target>, ...)` to `builds(<target>, ...)`"
            )

        target, *_pos_args = pos_args

        if isinstance(target, functools.partial):
            # partial'd args must come first, then user-specified args
            # otherwise, the parial'd args will take precedent, which
            # does not align with the behavior of partial itself
            _pos_args = list(target.args) + _pos_args
            kwargs_for_target = {**target.keywords, **kwargs_for_target}
            target = target.func

        BUILDS_ERROR_PREFIX = _utils.building_error_prefix(target)

        del pos_args

        zen_exclude: Union[Callable[[str], bool], Collection[Union[str, int]]] = (
            kwargs_for_target.pop("zen_exclude", frozenset())
        )
        zen_index_exclude: set[int] = set()

        if (
            not isinstance(zen_exclude, Collection) or isinstance(zen_exclude, str)
        ) and not callable(zen_exclude):
            raise TypeError(
                f"`zen_exclude` must be a non-string collection of strings and/or ints"
                f" or callable[[str], bool]. Got {zen_exclude}"
            )

        if isinstance(zen_exclude, Collection):
            _strings = []
            for item in zen_exclude:
                if isinstance(item, int):
                    zen_index_exclude.add(item)
                elif isinstance(item, str):
                    _strings.append(item)
                else:
                    raise TypeError(
                        f"`zen_exclude` must only contain ints or "
                        f"strings. Got {zen_exclude}"
                    )
            zen_exclude = frozenset(_strings).__contains__

        if not callable(target):
            raise TypeError(
                BUILDS_ERROR_PREFIX
                + "In `builds(<target>, ...), `<target>` must be callable/instantiable"
            )

        if not isinstance(populate_full_signature, bool):
            raise TypeError(
                f"`populate_full_signature` must be a boolean type, got: "
                f"{populate_full_signature}"
            )

        if zen_partial is not None and not isinstance(zen_partial, bool):
            raise TypeError(f"`zen_partial` must be a boolean type, got: {zen_partial}")

        _utils.validate_hydra_options(
            hydra_recursive=hydra_recursive, hydra_convert=hydra_convert
        )

        if any(not (is_dataclass(_b) and isinstance(_b, type)) for _b in builds_bases):
            raise TypeError("All `build_bases` must be a tuple of dataclass types")

        if zen_meta is None:
            zen_meta = {}

        if not isinstance(zen_meta, Mapping):
            raise TypeError(
                f"`zen_meta` must be a mapping (e.g. a dictionary), got: {zen_meta}"
            )

        if any(not isinstance(_key, str) for _key in zen_meta):
            raise TypeError(
                f"`zen_meta` must be a mapping whose keys are strings, got key(s):"
                f" {','.join(str(_key) for _key in zen_meta if not isinstance(_key, str))}"
            )

        target_path: str
        if manual_target_path is None:
            if (
                zen_convert_settings["flat_target"]
                and isinstance(target, type)
                and is_builds(target)
                and is_dataclass(target)
            ):
                # pass through _target_ field
                target_path = get_target_path(target)
                assert isinstance(target_path, str)
            else:
                target_path = cls._get_obj_path(target)
        else:
            target_path = manual_target_path

        if zen_wrappers is not None:
            if not isinstance(zen_wrappers, Sequence) or isinstance(zen_wrappers, str):
                zen_wrappers = (zen_wrappers,)

            validated_wrappers: Sequence[Union[str, Builds[Any]]] = []
            for wrapper in zen_wrappers:
                if wrapper is None:
                    continue
                # We are intentionally keeping each condition branched
                # so that test-coverage will be checked for each one
                if isinstance(wrapper, functools.partial):
                    wrapper = ZEN_VALUE_CONVERSION[functools.partial](
                        wrapper, CBuildsFn=cls
                    )

                if is_builds(wrapper):
                    # If Hydra's locate function starts supporting importing literals
                    # – or if we decide to ship our own locate function –
                    # then we should get the target of `wrapper` and make sure it is callable
                    if is_just(wrapper):
                        # `zen_wrappers` handles importing string; we can
                        # eliminate the indirection of Just and "flatten" this
                        # config
                        validated_wrappers.append(
                            safe_getattr(wrapper, JUST_FIELD_NAME)
                        )
                    else:
                        if hydra_recursive is False:
                            warnings.warn(
                                "A structured config was supplied for `zen_wrappers`. Its parent config has "
                                "`hydra_recursive=False`.\n If this value is not toggled to `True`, the config's "
                                "instantiation will result in an error"
                            )
                        validated_wrappers.append(wrapper)

                elif callable(wrapper):
                    validated_wrappers.append(cls._get_obj_path(wrapper))

                elif isinstance(wrapper, str):
                    # Assumed that wrapper is either a valid omegaconf-style interpolation string
                    # or a "valid" path for importing an object. The latter seems hopeless for validating:
                    # https://stackoverflow.com/a/47538106/6592114
                    # so we can't make any assurances here.
                    validated_wrappers.append(wrapper)
                else:
                    raise TypeError(
                        f"`zen_wrappers` requires a callable, targeted config, or a string, got: {wrapper}"
                    )

            del zen_wrappers
            validated_wrappers = tuple(validated_wrappers)
        else:
            validated_wrappers = ()

        # Check for reserved names
        for _name in chain(kwargs_for_target, zen_meta):
            if _name in HYDRA_FIELD_NAMES:
                err_msg = f"The field-name specified via `builds(..., {_name}=<...>)` is reserved by Hydra."
                if _name != TARGET_FIELD_NAME:
                    raise ValueError(
                        err_msg
                        + f" You can set this parameter via `builds(..., hydra_{_name[1:-1]}=<...>)`"
                    )
                else:
                    raise ValueError(err_msg)
            if _name.startswith(("hydra_", "_zen_", "zen_")):
                raise ValueError(
                    f"The field-name specified via `{_name}=<...>` is reserved by hydra-zen."
                    " You can manually create a dataclass to utilize this name in a structured config."
                )

        # list[tuple[str, type] | tuple[str, type, Any]]
        target_field: list[Union[tuple[str, Any], tuple[str, Any, Any]]]

        # zen_partial behavior:
        #
        # If zen_partial is not None: zen_partial dictates if output is PartialBuilds
        #
        # If zen_partial is None:
        #   - closest parent with partial-flag specified determines
        #     if output is PartialBuilds
        #   - if no parent, output is Builds
        #
        # If _partial_=True is inherited but zen-processing is used
        #    then set _partial_=False, _zen_partial=zen_partial
        #
        base_hydra_partial: Optional[bool] = (
            None  # state of closest parent with _partial_
        )
        base_zen_partial: Optional[bool] = (
            None  # state of closest parent with _zen_partial
        )

        # reflects state of closest parent that has partial field specified
        parent_partial: Optional[bool] = None

        for base in builds_bases:
            _set_this_iteration = False
            if base_hydra_partial is None:
                base_hydra_partial = safe_getattr(base, PARTIAL_FIELD_NAME, None)
                if parent_partial is None:
                    parent_partial = base_hydra_partial
                    _set_this_iteration = True

            if base_zen_partial is None:
                base_zen_partial = safe_getattr(base, ZEN_PARTIAL_FIELD_NAME, None)
                if parent_partial is None or (
                    _set_this_iteration and base_zen_partial is not None
                ):
                    parent_partial = parent_partial or base_zen_partial

            del _set_this_iteration

        if zen_partial is None:
            # zen_partial is inherited
            zen_partial = parent_partial

        del parent_partial

        requires_partial_field = zen_partial is not None

        requires_zen_processing: Final[bool] = (
            bool(zen_meta)
            or bool(validated_wrappers)
            or any(uses_zen_processing(b) for b in builds_bases)
        )

        if base_zen_partial:
            assert requires_zen_processing

        del base_zen_partial

        if not requires_zen_processing and requires_partial_field:
            target_field = [
                (
                    TARGET_FIELD_NAME,
                    str,
                    _utils.field(default=target_path, init=False, repr=target_repr),
                ),
                (
                    PARTIAL_FIELD_NAME,
                    bool,
                    _utils.field(default=bool(zen_partial), init=False),
                ),
            ]
        elif requires_zen_processing:
            # target is `hydra_zen.funcs.zen_processing`
            target_field = [
                (
                    TARGET_FIELD_NAME,
                    str,
                    _utils.field(default=ZEN_PROCESSING_LOCATION, init=False),
                ),
                (
                    ZEN_TARGET_FIELD_NAME,
                    str,
                    _utils.field(default=target_path, init=False),
                ),
            ]

            if requires_partial_field:
                target_field.append(
                    (
                        ZEN_PARTIAL_FIELD_NAME,
                        bool,
                        _utils.field(default=bool(zen_partial), init=False),
                    ),
                )
                if base_hydra_partial:
                    # Must explicitly set _partial_=False to prevent inheritance
                    target_field.append(
                        (
                            PARTIAL_FIELD_NAME,
                            bool,
                            _utils.field(default=False, init=False),
                        ),
                    )

            if zen_meta:
                target_field.append(
                    (
                        META_FIELD_NAME,
                        tuple[str, ...],
                        _utils.field(default=tuple(zen_meta), init=False),
                    ),
                )

            if validated_wrappers:
                if zen_meta:
                    # Check to see
                    tuple(
                        _utils.check_suspicious_interpolations(
                            validated_wrappers, zen_meta=zen_meta, target=target
                        )
                    )
                if len(validated_wrappers) == 1:
                    # we flatten the config to avoid unnecessary list
                    target_field.append(
                        (
                            ZEN_WRAPPERS_FIELD_NAME,
                            Union[
                                Union[str, Builds[Any]],
                                tuple[Union[str, Builds[Any]], Any],
                            ],
                            _utils.field(default=validated_wrappers[0], init=False),
                        ),
                    )
                else:
                    target_field.append(
                        (
                            ZEN_WRAPPERS_FIELD_NAME,
                            Union[
                                Union[str, Builds[Any]],
                                tuple[Union[str, Builds[Any]], Any],
                            ],
                            _utils.field(default=validated_wrappers, init=False),
                        ),
                    )
        else:
            target_field = [
                (
                    TARGET_FIELD_NAME,
                    str,
                    _utils.field(default=target_path, init=False, repr=target_repr),
                )
            ]

        del base_hydra_partial
        del requires_partial_field

        base_fields = target_field

        if hydra_recursive is not None:
            base_fields.append(
                (
                    RECURSIVE_FIELD_NAME,
                    bool,
                    _utils.field(default=hydra_recursive, init=False),
                )
            )

        if hydra_convert is not None:
            base_fields.append(
                (
                    CONVERT_FIELD_NAME,
                    str,
                    _utils.field(default=hydra_convert, init=False),
                )
            )

        if hydra_defaults is not None:
            if not _utils.valid_defaults_list(hydra_defaults):
                raise HydraZenValidationError(
                    f"`hydra_defaults` must be type `None | list[str | dict[str, str | list[str] | None ]]`"
                    f", Got: {repr(hydra_defaults)}"
                )
            hydra_defaults = cls._sanitize_collection(
                hydra_defaults, convert_dataclass=False
            )
            base_fields.append(
                (
                    DEFAULTS_LIST_FIELD_NAME,
                    list[Any],
                    _utils.field(
                        default_factory=lambda: list(hydra_defaults),
                        init=False,
                    ),
                )
            )

        if _pos_args:
            base_fields.append(
                (
                    POS_ARG_FIELD_NAME,
                    tuple[Any, ...],
                    _utils.field(
                        default=tuple(
                            cls._make_hydra_compatible(
                                x,
                                error_prefix=BUILDS_ERROR_PREFIX,
                                convert_dataclass=zen_convert_settings["dataclass"],
                            )
                            for x in _pos_args
                        ),
                        init=False,
                    ),
                )
            )

        _sig_target = cls._get_sig_obj(target)
        pydantic = sys.modules.get("pydantic")
        try:
            # We want to rely on `inspect.signature` logic for raising
            # against an uninspectable sig, before we start inspecting
            # class-specific attributes below.
            signature_params = dict(inspect.signature(target).parameters)  # type: ignore
        except ValueError:
            if populate_full_signature:
                raise ValueError(
                    BUILDS_ERROR_PREFIX
                    + f"{target} does not have an inspectable signature. "
                    f"`builds({_utils.safe_name(target)}, populate_full_signature=True)` is not supported"
                )
            signature_params: dict[str, inspect.Parameter] = {}
            # We will turn off signature validation for objects that didn't have
            # a valid signature. This will enable us to do things like `build(dict, a=1)`
            target_has_valid_signature: bool = False
        else:
            # Dealing with this bug: https://bugs.python.org/issue40897
            #
            # In Python < 3.9.1, `inspect.signature will look first to
            # any implementation __new__, even if it is inherited and if
            # there is a "fresher" __init__.
            #
            # E.g. anything that inherits from `typing.Generic` and
            # does not implement its own __new__ will have a reported sig
            # of (*args, **kwargs)
            #
            # This looks specifically for the scenario that the target
            # has inherited from a parent that implements __new__ and
            # the target implements only __init__.

            if pydantic is not None and (
                _sig_target is pydantic.BaseModel.__init__
                # pydantic v2.0
                or is_dataclass(target)
                and hasattr(target, "__pydantic_config__")
            ):
                pass
            elif _sig_target is not target:
                _params = tuple(inspect.signature(_sig_target).parameters.items())

                if (
                    _params and _params[0][1].kind is not _VAR_POSITIONAL
                ):  # pragma: no cover
                    # Exclude self/cls
                    #
                    # There are weird edge cases
                    # where the first arg is *args, not self.
                    _params = _params[1:]
                else:  # pragma: no cover
                    pass

                signature_params = {k: v for k, v in _params}
                del _params

            target_has_valid_signature: bool = True

        if is_dataclass(target) or (
            pydantic is not None
            and isinstance(target, type)
            and issubclass(target, pydantic.BaseModel)
        ):
            # Update `signature_params` so that any param with `default=<factory>`
            # has its default replaced with `<factory>()`
            # If this is a mutable value, `builds` will automatically re-pack
            # it using a default factory
            if is_dataclass(target):
                _fields = {f.name: f for f in fields(target)}
            else:
                _fields = target.__fields__  # type: ignore
            _update = {}
            for name, param in signature_params.items():
                if name not in _fields:
                    # field is InitVar
                    continue
                f = _fields[name]
                if f.default_factory is not MISSING and f.default_factory is not None:
                    _update[name] = inspect.Parameter(
                        name,
                        param.kind,
                        annotation=param.annotation,
                        default=f.default_factory(),
                    )
            signature_params.update(_update)
            if (
                zen_convert_settings["flat_target"]
                and TARGET_FIELD_NAME in signature_params
            ):
                signature_params.pop(TARGET_FIELD_NAME)
            del _update
            del _fields

        # `get_type_hints` properly resolves forward references, whereas annotations from
        # `inspect.signature` do not
        try:
            type_hints = get_type_hints(_sig_target)

            del _sig_target
            # We don't need to pop self/class because we only make on-demand
            # requests from `type_hints`

        except (
            TypeError,  # ufuncs, which do not have inspectable type hints
            NameError,  # Unresolvable forward reference
            AttributeError,  # Class doesn't have "__new__" or "__init__"
        ):
            type_hints: dict[str, Any] = {}

        sig_by_kind: dict[Any, list[inspect.Parameter]] = {
            _POSITIONAL_ONLY: [],
            _POSITIONAL_OR_KEYWORD: [],
            _VAR_POSITIONAL: [],
            _KEYWORD_ONLY: [],
            _VAR_KEYWORD: [],
        }

        for p in signature_params.values():
            sig_by_kind[p.kind].append(p)

        # these are the names of the only parameters in the signature of `target` that can
        # be referenced by name
        nameable_params_in_sig: set[str] = {
            p.name
            for p in chain(
                sig_by_kind[_POSITIONAL_OR_KEYWORD], sig_by_kind[_KEYWORD_ONLY]
            )
        }

        if not _pos_args and builds_bases:
            # pos_args is potentially inherited
            for _base in builds_bases:
                _pos_args = safe_getattr(_base, POS_ARG_FIELD_NAME, ())

                # validates
                _pos_args = tuple(
                    cls._make_hydra_compatible(
                        x, allow_zen_conversion=False, convert_dataclass=False
                    )
                    for x in _pos_args
                )
                if _pos_args:
                    break

        fields_set_by_bases: set[str] = {
            _field.name
            for _base in builds_bases
            for _field in fields(_base)
            if _field.name not in HYDRA_FIELD_NAMES
            and not _field.name.startswith("_zen_")
        }

        # Validate that user-specified arguments satisfy target's signature.
        # Should catch:
        #    - bad parameter names
        #    - too many parameters-by-position
        #    - multiple values specified for parameter (via positional and by-name)
        #
        # We don't raise on an under-specified signature because it is possible that the
        # resulting dataclass will simply be inherited from and extended.
        # The issues we catch here cannot be fixed downstream.
        if target_has_valid_signature:
            if not sig_by_kind[_VAR_KEYWORD]:
                # check for unexpected kwargs
                if not set(kwargs_for_target) <= nameable_params_in_sig:
                    _unexpected = set(kwargs_for_target) - nameable_params_in_sig
                    raise TypeError(
                        BUILDS_ERROR_PREFIX
                        + f"The following unexpected keyword argument(s) was specified for {target_path} "
                        f"via `builds`: {', '.join(_unexpected)}"
                    )
                if not fields_set_by_bases <= nameable_params_in_sig and not (
                    fields_set_by_bases - nameable_params_in_sig
                ) <= set(zen_meta):
                    # field inherited by base is not present in sig
                    # AND it is not excluded via `zen_meta`
                    _unexpected = fields_set_by_bases - nameable_params_in_sig
                    raise TypeError(
                        BUILDS_ERROR_PREFIX
                        + f"The following unexpected keyword argument(s) for {target_path} "
                        f"was specified via inheritance from a base class: "
                        f"{', '.join(_unexpected)}"
                    )

            if _pos_args:
                named_args = set(kwargs_for_target).union(fields_set_by_bases)

                # indicates that number of parameters that could be specified by name,
                # but are specified by position
                _num_nameable_args_by_position = max(
                    0, len(_pos_args) - len(sig_by_kind[_POSITIONAL_ONLY])
                )
                if named_args:
                    # check for multiple values for arg, specified both via positional and kwarg
                    # E.g.: def f(x, y): ...
                    # f(1, 2, y=3)  # multiple values for `y`
                    for param in sig_by_kind[_POSITIONAL_OR_KEYWORD][
                        :_num_nameable_args_by_position
                    ]:
                        if param.name in named_args:
                            raise TypeError(
                                BUILDS_ERROR_PREFIX
                                + f"Multiple values for argument {param.name} were specified for "
                                f"{target_path} via `builds`"
                            )
                if not sig_by_kind[
                    _VAR_POSITIONAL
                ] and _num_nameable_args_by_position > len(
                    sig_by_kind[_POSITIONAL_OR_KEYWORD]
                ):
                    # Too many positional args specified.
                    # E.g.: def f(x, y): ...
                    # f(1, 2, 3)
                    _num_positional = len(sig_by_kind[_POSITIONAL_ONLY]) + len(
                        sig_by_kind[_POSITIONAL_OR_KEYWORD]
                    )
                    _num_with_default = sum(
                        p.default is not inspect.Parameter.empty
                        and p.kind is _POSITIONAL_OR_KEYWORD
                        for p in signature_params.values()
                    )
                    _permissible = (
                        f"{_num_positional}"
                        if not _num_with_default
                        else f"{_num_positional - _num_with_default} to {_num_positional}"
                    )
                    raise TypeError(
                        BUILDS_ERROR_PREFIX
                        + f"{target_path} takes {_permissible} positional args, but "
                        f"{len(_pos_args)} were specified via `builds`"
                    )

        # Create valid dataclass fields from the user-specified values
        #
        # user_specified_params: arg-name -> (arg-name, arg-type, field-w-value)
        #  - arg-type: taken from the parameter's annotation in the target's signature
        #    and is resolved to one of the type annotations supported by hydra if possible,
        #    otherwise, is Any
        #  - arg-value: mutable values are automatically specified using default-factory
        user_specified_named_params: dict[str, tuple[str, type, Any]] = {
            name: (name, type_hints.get(name, Any), value)
            for name, value in kwargs_for_target.items()
            if not zen_exclude(name)
        }

        # support negative indices
        zen_index_exclude = {ind % len(signature_params) for ind in zen_index_exclude}

        if populate_full_signature is True:
            # Populate dataclass fields based on the target's signature.
            #
            # A user-specified parameter value (via `kwargs_for_target`) takes precedent over
            # the default value from the signature

            # Fields with default values must come after those without defaults,
            # so we will collect these as we loop through the parameters and
            # add them to the fields at the end.
            #
            # Parameter ordering should only differ from the target's signature
            # if the user specified a value for a parameter that had no default
            _fields_with_default_values: list[Field_Entry] = []

            # we need to keep track of what user-specified params we have set
            _seen: set[str] = set()

            for n, param in enumerate(signature_params.values()):
                if n in zen_index_exclude or zen_exclude(param.name):
                    continue

                if n + 1 <= len(_pos_args):
                    # Positional parameters are populated from "left to right" in the signature.
                    # We have already done validation, so we know that positional params aren't redundant
                    # with named params (including inherited params).
                    continue

                if param.name not in nameable_params_in_sig:
                    # parameter cannot be specified by name
                    continue

                if param.name in user_specified_named_params:
                    # user-specified parameter is preferred
                    _fields_with_default_values.append(
                        user_specified_named_params[param.name]
                    )
                    _seen.add(param.name)
                else:
                    # any parameter whose default value is None is automatically
                    # annotated with `Optional[...]`. This improves flexibility with
                    # Hydra's type-validation
                    param_field = (
                        param.name,
                        type_hints.get(param.name, Any),
                    )

                    if param.default is inspect.Parameter.empty:
                        if not zen_partial:
                            # No default value specified in signature or by the user.
                            # We don't include these fields if the user specified a partial build
                            # because we assume that they want to fill these in by using partial
                            base_fields.append(param_field)
                    else:
                        param_field += (param.default,)
                        _fields_with_default_values.append(param_field)

            base_fields.extend(_fields_with_default_values)

            if sig_by_kind[_VAR_KEYWORD]:
                # if the signature has **kwargs, then we need to add any user-specified
                # parameters that have not already been added
                base_fields.extend(
                    entry
                    for name, entry in user_specified_named_params.items()
                    if name not in _seen
                )
        else:
            base_fields.extend(user_specified_named_params.values())

        if zen_meta:
            _meta_names = set(zen_meta)

            if _meta_names & nameable_params_in_sig:
                raise ValueError(
                    f"`builds(..., zen_meta=<...>)`: `zen_meta` cannot not specify "
                    f"names that exist in the target's signature: "
                    f"{','.join(_meta_names & nameable_params_in_sig)}"
                )

            if _meta_names & set(user_specified_named_params):
                raise ValueError(
                    f"`builds(..., zen_meta=<...>)`: `zen_meta` cannot not specify "
                    f"names that are common with those specified in **kwargs_for_target: "
                    f"{','.join(_meta_names & set(user_specified_named_params))}"
                )

            # We don't check for collisions between `zen_meta` names and the
            # names of inherited fields. Thus `zen_meta` can effectively be used
            # to "delete" names from a config, via inheritance.
            base_fields.extend((name, Any, value) for name, value in zen_meta.items())

        if dataclass_name is None:
            if zen_partial is not True:
                dataclass_name = f"Builds_{_utils.safe_name(target)}"
            else:
                dataclass_name = f"PartialBuilds_{_utils.safe_name(target)}"

        # validate that fields set via bases are OK; cannot perform zen-casting
        # on fields
        for base in builds_bases:
            for field_ in fields(base):
                if field_.default is not MISSING:
                    # performs validation
                    cls._make_hydra_compatible(
                        field_.default,
                        allow_zen_conversion=False,
                        error_prefix=BUILDS_ERROR_PREFIX,
                        field_name=field_.name + " (set via inheritance)",
                        convert_dataclass=False,
                    )
                del field_

        # sanitize all types and configured values
        sanitized_base_fields: list[
            Union[tuple[str, Any], tuple[str, Any, Field[Any]]]
        ] = []

        for item in base_fields:
            name = item[0]
            type_ = item[1]
            if len(item) == 2:
                sanitized_base_fields.append((name, cls._sanitized_type(type_)))
            else:
                assert len(item) == 3, item
                value = item[-1]

                if not isinstance(value, _Field):
                    _field = cls._sanitized_field(
                        value,
                        error_prefix=BUILDS_ERROR_PREFIX,
                        field_name=item[0],
                        convert_dataclass=zen_convert_settings["dataclass"],
                    )
                else:
                    _field = value

                # If `.default` is not set, then `value` is a Hydra-supported mutable
                # value, and thus it is "sanitized"
                sanitized_value = safe_getattr(_field, "default", value)
                sanitized_type = (
                    cls._sanitized_type(type_, wrap_optional=sanitized_value is None)
                    if _retain_type_info(
                        type_=type_,
                        value=sanitized_value,
                        hydra_recursive=hydra_recursive,
                    )
                    else Any
                )
                sanitized_base_fields.append((name, sanitized_type, _field))
                del value
                del _field
                del sanitized_value

        dataclass_options["cls_name"] = dataclass_name
        dataclass_options["bases"] = builds_bases
        assert _utils.parse_strict_dataclass_options(dataclass_options)

        out = make_dataclass(fields=sanitized_base_fields, **dataclass_options)

        if module is not None:
            out.__module__ = module

        out.__doc__ = (
            f"A structured config designed to {'partially ' if zen_partial else ''}"
            f"initialize/call `{target_path}` upon instantiation by hydra."
        )
        if hasattr(target, "__doc__"):  # pragma: no branch
            target_doc = target.__doc__
            if target_doc:
                out.__doc__ += (
                    f"\n\nThe docstring for {_utils.safe_name(target)} :\n\n"
                    + target_doc
                )

        assert requires_zen_processing is uses_zen_processing(out)

        # _partial_=True should never be relied on when zen-processing is being used.
        assert not (
            requires_zen_processing and safe_getattr(out, PARTIAL_FIELD_NAME, False)
        )

        return cast(
            Union[type[Builds[Importable]], type[BuildsWithSig[type[R], P]]], out
        )

    @overload
    @classmethod
    def just(
        cls,
        obj: TP,
        *,
        zen_convert: Optional[ZenConvert] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> TP: ...

    @overload
    @classmethod
    def just(
        cls,
        obj: complex,
        *,
        zen_convert: Optional[ZenConvert] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> "ConfigComplex": ...

    @overload
    @classmethod
    def just(
        cls,
        obj: TC,
        *,
        zen_convert: Optional[ZenConvert] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> JustT[TC]: ...

    @overload
    @classmethod
    def just(
        cls,
        obj: TB,
        *,
        zen_convert: Optional[ZenConvert] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> Builds[type[TB]]: ...

    @overload
    @classmethod
    def just(
        cls,
        obj: TD,
        *,
        zen_convert: Literal[None] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> type[Builds[type[TD]]]: ...

    @overload
    @classmethod
    def just(
        cls,
        obj: DataClass_,
        *,
        zen_convert: ZenConvert,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> Any: ...

    @overload
    @classmethod
    def just(
        cls,
        obj: Any,
        *,
        zen_convert: Optional[ZenConvert] = ...,
        hydra_recursive: Optional[bool] = ...,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
        zen_dataclass: Optional[DataclassOptions] = ...,
    ) -> Any: ...

    @classmethod
    def just(
        cls,
        obj: Any,
        *,
        zen_convert: Optional[ZenConvert] = None,
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
    ) -> Any:
        """`just(obj)` returns a config that, when instantiated, just returns `obj`.

        I.e., `instantiate(just(obj)) == obj`

        `just` is designed to be idempotent. I.e., `just(obj) == just(just(obj))`

        See the docstring for `hydra_zen.just`
        """
        convert_settings = merge_settings(zen_convert, _JUST_CONVERT_SETTINGS)
        del zen_convert
        _utils.validate_hydra_options(
            hydra_recursive=hydra_recursive, hydra_convert=hydra_convert
        )
        if zen_dataclass is None:
            zen_dataclass = {}

        return cls._make_hydra_compatible(
            obj,
            allow_zen_conversion=True,
            structured_conf_permitted=True,
            field_name="",
            error_prefix="",
            convert_dataclass=convert_settings["dataclass"],
            hydra_convert=hydra_convert,
            hydra_recursive=hydra_recursive,
            zen_dataclass=_utils.parse_dataclass_options(zen_dataclass),
        )

    @classmethod
    def make_config(
        cls,
        *fields_as_args: Union[str, ZenField],
        hydra_recursive: Optional[bool] = None,
        hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
        hydra_defaults: Optional[DefaultsList] = None,
        zen_dataclass: Optional[DataclassOptions] = None,
        bases: tuple[type[DataClass_], ...] = (),
        zen_convert: Optional[ZenConvert] = None,
        **fields_as_kwargs: Union[T, ZenField],
    ) -> type[DataClass]:
        """
        Returns a config with user-defined field names and, optionally,
        associated default values and/or type annotations.

        See the docstring for hydra_zen.make_config
        """
        convert_settings = _utils.merge_settings(zen_convert, _MAKE_CONFIG_SETTINGS)
        convert_settings = cast(ZenConvert, convert_settings)
        del zen_convert

        if zen_dataclass is None:
            zen_dataclass = {}

        # initial validation
        _utils.parse_dataclass_options(zen_dataclass)

        if "frozen" in fields_as_kwargs:
            warnings.warn(
                HydraZenDeprecationWarning(
                    "Specifying `builds(frozen=<...>)` is deprecated. Instead, "
                    "specify `builds(zen_dataclass={'frozen': <...>})"
                ),
                stacklevel=2,
            )
            zen_dataclass["frozen"] = fields_as_kwargs.pop("frozen")  # type: ignore

        if "config_name" in fields_as_kwargs:
            warnings.warn(
                HydraZenDeprecationWarning(
                    "Specifying `make_config(config_name=<...>)` is deprecated. "
                    "Instead specify `make_config(zen_dataclass={'cls_name': <...>})"
                ),
                stacklevel=2,
            )
            zen_dataclass["cls_name"] = fields_as_kwargs.pop("config_name")  # type: ignore

        if not bases:
            bases = zen_dataclass.get("bases", ())

        zen_dataclass.setdefault("cls_name", "Config")
        dataclass_options = _utils.parse_dataclass_options(zen_dataclass)

        for _field in fields_as_args:
            if not isinstance(_field, (str, ZenField)):
                raise TypeError(
                    f"`fields_as_args` can only consist of field-names (i.e. strings) or "
                    f"`ZenField` instances. Got: "
                    f"{', '.join(str(x) for x in fields_as_args if not isinstance(x, (str, ZenField)))}"
                )
            if isinstance(_field, ZenField) and _field.name is NOTHING:
                raise ValueError(
                    f"All `ZenField` instances specified in `fields_as_args` must have a "
                    f"name associated with it. Got: {_field}"
                )
        for name, _field in fields_as_kwargs.items():
            if isinstance(_field, ZenField):
                if _field.name is not NOTHING and _field.name != name:
                    raise ValueError(
                        f"`fields_as_kwargs` specifies conflicting names: the kwarg {name} "
                        f"is associated with a `ZenField` with name {_field.name}"
                    )
                else:
                    _field.name = name

        if fields_as_args:
            all_names = [
                f.name if isinstance(f, ZenField) else f for f in fields_as_args
            ]
            all_names.extend(fields_as_kwargs)

            if len(all_names) != len(set(all_names)):
                raise ValueError(
                    f"`fields_as_args` cannot specify the same field-name multiple times."
                    f" Got multiple entries for:"
                    f" {', '.join(str(n) for n, count in Counter(all_names).items() if count > 1)}"
                )
            for _name in all_names:
                if isinstance(_name, str) and _name.startswith("_zen_"):
                    raise ValueError(
                        f"The field-name specified via `{_name}=<...>` is reserved by hydra-zen."
                        " You can manually create a dataclass to utilize this name in a structured config."
                    )
            del all_names

        if "defaults" in fields_as_kwargs:
            if hydra_defaults is not None:
                raise TypeError(
                    "`defaults` and `hydra_defaults` cannot be specified simultaneously"
                )
            _defaults = fields_as_kwargs.pop("defaults")

            if not isinstance(_defaults, ZenField):  # pragma: no branch
                hydra_defaults = _defaults  # type: ignore

        # validate hydra-args via `builds`
        # also check for use of reserved names
        _tmp: Any = None

        cls.builds(
            dict,
            hydra_convert=hydra_convert,
            hydra_recursive=hydra_recursive,
            hydra_defaults=hydra_defaults,
            **{k: _tmp for k in fields_as_kwargs},
        )

        normalized_fields: dict[str, ZenField] = {}

        for _field in fields_as_args:
            if isinstance(_field, str):
                normalized_fields[_field] = ZenField(
                    name=_field,
                    hint=Any,
                    zen_convert=convert_settings,
                    _builds_fn=cls,
                )
            else:
                assert isinstance(_field.name, str)
                normalized_fields[_field.name] = _field

        for name, value in fields_as_kwargs.items():
            if not isinstance(value, ZenField):
                normalized_fields[name] = ZenField(
                    name=name,
                    default=value,
                    zen_convert=convert_settings,
                    _builds_fn=cls,
                )
            else:
                normalized_fields[name] = value

        # fields without defaults must come first
        config_fields: list[Union[tuple[str, type], tuple[str, type, Any]]] = [
            (str(f.name), f.hint)
            for f in normalized_fields.values()
            if f.default is NOTHING
        ]

        config_fields.extend(
            [  # type: ignore
                (
                    str(f.name),
                    (
                        # f.default: Field
                        # f.default.default: Any
                        f.hint
                        if _retain_type_info(
                            type_=f.hint,
                            value=f.default.default,
                            hydra_recursive=hydra_recursive,
                        )
                        else Any
                    ),
                    f.default,
                )
                for f in normalized_fields.values()
                if f.default is not NOTHING
            ]
        )

        if hydra_recursive is not None:
            config_fields.append(
                (
                    RECURSIVE_FIELD_NAME,
                    bool,
                    _utils.field(default=hydra_recursive, init=False),
                )
            )

        if hydra_convert is not None:
            config_fields.append(
                (
                    CONVERT_FIELD_NAME,
                    str,
                    _utils.field(default=hydra_convert, init=False),
                )
            )

        if hydra_defaults is not None:
            hydra_defaults = cls._sanitize_collection(
                hydra_defaults, convert_dataclass=False
            )
            config_fields.append(
                (
                    DEFAULTS_LIST_FIELD_NAME,
                    list[Any],
                    _utils.field(
                        default_factory=lambda: list(hydra_defaults), init=False
                    ),
                )
            )

        dataclass_options["bases"] = bases
        module = dataclass_options.pop("module", None)
        assert _utils.parse_strict_dataclass_options(
            dataclass_options
        ), dataclass_options

        out = make_dataclass(fields=config_fields, **dataclass_options)

        if module is not None:
            out.__module__ = module

        if hasattr(out, ZEN_TARGET_FIELD_NAME) and not uses_zen_processing(out):
            raise ValueError(
                f"{out.__name__} inherits from base classes that overwrite some fields "
                f"associated with zen-processing features. As a result, this config will "
                f"not instantiate correctly."
            )
        if safe_getattr(out, PARTIAL_FIELD_NAME, False) and uses_zen_processing(out):
            raise ValueError(
                f"{out.__name__} specifies both `{PARTIAL_FIELD_NAME}=True` and `"
                f"{ZEN_PARTIAL_FIELD_NAME}=True`. This config will not instantiate "
                f"correctly. This is typically caused by inheriting from multiple, "
                f"conflicting configs."
            )

        return cast(type[DataClass], out)

    # cover zen_exclude=() -> (1, 2, 3)
    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[P, Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: tuple[()],
    ) -> type[BuildsWithSig[type[dict[str, Any]], P]]: ...

    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[Concatenate[Any, P], Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: tuple[Literal[0]],
    ) -> type[BuildsWithSig[type[dict[str, Any]], P]]: ...

    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[Concatenate[Any, Any, P], Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: tuple[Literal[0], Literal[1]],
    ) -> type[BuildsWithSig[type[dict[str, Any]], P]]: ...

    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[Concatenate[Any, Any, Any, P], Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: tuple[Literal[0], Literal[1], Literal[2]],
    ) -> type[BuildsWithSig[type[dict[str, Any]], P]]: ...

    # no zen-exclude

    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[P, Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: Literal[None] = ...,
    ) -> type[BuildsWithSig[type[dict[str, Any]], P]]: ...

    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[P, Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: Union["Collection[Union[str, int]]", Callable[[str], bool]],
    ) -> type[Builds[type[dict[str, Any]]]]: ...

    @overload
    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Callable[P, Any],
        *,
        zen_dataclass: Optional[DataclassOptions] = ...,
        zen_exclude: Union[
            None, "Collection[Union[str, int]]", Callable[[str], bool]
        ] = ...,
        **kwarg_overrides: T,
    ) -> type[Builds[type[dict[str, Any]]]]: ...

    @classmethod
    def kwargs_of(
        cls: type[Self],
        __hydra_target: Union[
            Callable[P, Any],
            Callable[Concatenate[Any, P], Any],
            Callable[Concatenate[Any, Any, P], Any],
            Callable[Concatenate[Any, Any, Any, P], Any],
        ],
        *,
        zen_dataclass: Optional[DataclassOptions] = None,
        zen_exclude: Union[
            None, "Collection[Union[str, int]]", Callable[[str], bool]
        ] = None,
        **kwarg_overrides: T,
    ) -> Union[
        type[BuildsWithSig[type[dict[str, Any]], P]], type[Builds[type[dict[str, Any]]]]
    ]:
        """Returns a config whose signature matches that of the provided target.

        Instantiating the config returns a dictionary.

        Parameters
        ----------
        __hydra_target : Callable[P, Any]
            An object with an inspectable signature.

        zen_exclude : Collection[str | int] | Callable[[str], bool], optional (default=[])
            Specifies parameter names and/or indices, or a function for checking names,
            to exclude those parameters from the config-creation process.

        **kwarg_overrides : T
            Named overrides for the parameters' default values.

        Returns
        -------
        type[Builds[type[dict[str, Any]]]]

        Examples
        --------
        >>> from inspect import signature
        >>> from hydra_zen import kwargs_of, instantiate

        >>> Config = kwargs_of(lambda x, y: None)
        >>> signature(Config)
        <Signature (x:Any, y: Any) -> None>
        >>> config = Config(x=1, y=2)
        >>> config
        kwargs_of_lambda(x=1, y=2)
        >>> instantiate(config)
        {'x': 1, 'y': 2}

        Excluding the first parameter from the target's signature:

        >>> Config = kwargs_of(lambda *, x, y: None, zen_exclude=(0,))
        >>> signature(Config)  # note: type checkers sees that x is removed as well
        <Signature (y: Any) -> None>
        >>> instantiate(Config(y=88))
        {'y': 88}


        Overwriting a default

        >>> Config = kwargs_of(lambda *, x, y: None, y=22)
        >>> signature(Config)
        <Signature (x: Any, y: Any = 22) -> None>
        """
        base_zen_detaclass: DataclassOptions = (
            cls._default_dataclass_options_for_kwargs_of.copy()
            if cls._default_dataclass_options_for_kwargs_of
            else {}
        )
        if zen_dataclass is None:
            zen_dataclass = {}

        zen_dataclass = {**base_zen_detaclass, **zen_dataclass}
        zen_dataclass["target"] = "builtins.dict"
        zen_dataclass.setdefault(
            "cls_name", f"kwargs_of_{_utils.safe_name(__hydra_target)}"
        )
        zen_dataclass.setdefault("target_repr", False)

        if zen_exclude is None:
            zen_exclude = ()
        return cls.builds(  # type: ignore
            __hydra_target,
            populate_full_signature=True,
            zen_exclude=zen_exclude,  # type: ignore
            zen_dataclass=zen_dataclass,
            **kwarg_overrides,  # type: ignore
        )


class DefaultBuilds(BuildsFn[SupportedPrimitive]):
    _default_dataclass_options_for_kwargs_of = {}


builds: Final = DefaultBuilds.builds
kwargs_of: Final = DefaultBuilds.kwargs_of


@dataclass(unsafe_hash=True)
class ConfigComplex:
    real: Any
    imag: Any
    _target_: str = field(default=BuildsFn._get_obj_path(complex), init=False)
    CBuildsFn: InitVar[type[BuildsFn[Any]]]

    def __post_init__(self, CBuildsFn: type[BuildsFn[Any]]) -> None:
        del CBuildsFn


@dataclass(unsafe_hash=True)
class ConfigPath:
    _args_: tuple[str]
    _target_: str = field(default=BuildsFn._get_obj_path(Path), init=False)
    CBuildsFn: InitVar[type[BuildsFn[Any]]]

    def __post_init__(self, CBuildsFn: type[BuildsFn[Any]]) -> None:  # pragma: no cover
        del CBuildsFn


def get_target_path(obj: Union[HasTarget, HasTargetInst]) -> Any:
    """
    Returns the import-path from a targeted config.

    Parameters
    ----------
    obj : HasTarget
        An object with a ``_target_`` attribute.

    Returns
    -------
    target_str : str
        The import path stored on the config object.

    Raises
    ------
    TypeError: ``obj`` does not have a ``_target_`` attribute.
    """
    if is_old_partial_builds(obj):
        # obj._partial_target_ is `Just[obj]`
        return get_target(getattr(obj, "_partial_target_"))
    elif uses_zen_processing(obj):
        field_name = ZEN_TARGET_FIELD_NAME
    elif is_just(obj):
        field_name = JUST_FIELD_NAME
    elif is_builds(obj):
        field_name = TARGET_FIELD_NAME
    else:
        raise TypeError(
            f"`obj` must specify a target; i.e. it must have an attribute named"
            f" {TARGET_FIELD_NAME} or named {ZEN_PARTIAL_FIELD_NAME} that"
            f" points to a target-object or target-string"
        )
    target = safe_getattr(obj, field_name)
    return target


@overload
def get_target(obj: InstOrType[Builds[_T]]) -> _T: ...


@overload
def get_target(obj: HasTargetInst) -> Any: ...


@overload
def get_target(obj: HasTarget) -> Any: ...


def get_target(obj: Union[HasTarget, HasTargetInst]) -> Any:
    """
    Returns the target-object from a targeted config.

    Parameters
    ----------
    obj : HasTarget
        An object with a ``_target_`` attribute.

    Returns
    -------
    target : Any
        The target object of the config.

        Note that this will import the object using the import
        path specified by the config.

    Raises
    ------
    TypeError: ``obj`` does not have a ``_target_`` attribute.

    Examples
    --------
    **Basic usage**

    `get_target` works on all variety of configs produced by `builds` and
    `just`.

    >>> from hydra_zen import builds, just, get_target
    >>> get_target(builds(int))
    <class 'int'>
    >>> get_target(builds(int, zen_partial=True))
    <class 'int'>
    >>> get_target(just(str))
    <class 'str'>

    It works for manually-defined configs:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class A:
    ...     _target_: str = "builtins.dict"

    >>> get_target(A())
    <class 'dict'>

    and for configs loaded from yamls.

    >>> from hydra_zen import load_from_yaml, save_as_yaml

    >>> class B: pass
    >>> Conf = builds(B)

    >>> save_as_yaml(Conf, "config.yaml")
    >>> loaded_conf = load_from_yaml("config.yaml")

    Note that the target of ``loaded_conf`` can be accessed without
    instantiating the config.

    >>> get_target(loaded_conf)  # type: ignore
    __main__.B
    """
    target = get_target_path(obj=obj)

    if isinstance(target, str):
        target = get_obj(path=target)
    else:
        # Hydra 1.1.0 permits objects-as-_target_ instead of strings
        # https://github.com/facebookresearch/hydra/issues/1017
        pass  # makes sure we cover this branch in tests

    return target


def mutable_value(
    x: _T,
    *,
    zen_convert: Optional[ZenConvert] = None,
    BuildsFunction: type[BuildsFn[Any]] = BuildsFn[Any],
) -> _T:
    """Used to set a mutable object as a default value for a field
    in a dataclass.

    This is an alias for ``field(default_factory=lambda: type(x)(x))``

    Note that ``type(x)(...)`` serves to make a copy

    Examples
    --------
    >>> from hydra_zen import mutable_value
    >>> from dataclasses import dataclass

    See https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    >>> @dataclass  # doctest: +SKIP
    ... class HasMutableDefault:
    ...     a_list: list  = [1, 2, 3]  # error: mutable default

    Using `mutable_value` to specify the default list:

    >>> @dataclass
    ... class HasMutableDefault:
    ...     a_list: list  = mutable_value([1, 2, 3])  # ok

    >>> x = HasMutableDefault()
    >>> x.a_list.append(-1)  # does not append to `HasMutableDefault.a_list`
    >>> x
    HasMutableDefault(a_list=[1, 2, 3, -1])
    >>> HasMutableDefault()
    HasMutableDefault(a_list=[1, 2, 3])"""
    return BuildsFunction._mutable_value(x, zen_convert=zen_convert)


def convert_complex(
    value: complex, CBuildsFn: type[BuildsFn[Any]]
) -> Builds[type[complex]]:
    return cast(
        Builds[type[complex]],
        ConfigComplex(real=value.real, imag=value.imag, CBuildsFn=CBuildsFn),
    )


ZEN_VALUE_CONVERSION[complex] = convert_complex


if Path in ZEN_SUPPORTED_PRIMITIVES:  # pragma: no cover

    def convert_path(value: Path, CBuildsFn: type[BuildsFn[Any]]) -> Builds[type[Path]]:
        return cast(
            Builds[type[Path]], ConfigPath(_args_=(str(value),), CBuildsFn=CBuildsFn)
        )

    ZEN_VALUE_CONVERSION[Path] = convert_path
    ZEN_VALUE_CONVERSION[PosixPath] = convert_path
    ZEN_VALUE_CONVERSION[WindowsPath] = convert_path


def _unpack_partial(
    value: Partial[_T], CBuildsFn: type[BuildsFn[Any]]
) -> PartialBuilds[type[_T]]:
    target = cast(type[_T], value.func)
    return CBuildsFn.builds(target, *value.args, **value.keywords, zen_partial=True)()


@dataclass(unsafe_hash=True)
class ConfigFromTuple:
    _args_: tuple[Any, ...]
    _target_: str
    CBuildsFn: InitVar[type[BuildsFn[Any]]]

    def __post_init__(self, CBuildsFn: type[BuildsFn[Any]]) -> None:
        self._args_ = (
            CBuildsFn._make_hydra_compatible(
                tuple(self._args_),
                convert_dataclass=True,
                allow_zen_conversion=True,
                structured_conf_permitted=True,
            ),
        )


@dataclass(unsafe_hash=True)
class ConfigFromDict:
    _args_: Any
    _target_: str
    CBuildsFn: InitVar[type[BuildsFn[Any]]]

    def __post_init__(self, CBuildsFn: type[BuildsFn[Any]]) -> None:
        self._args_ = (
            CBuildsFn._make_hydra_compatible(
                dict(self._args_),
                convert_dataclass=True,
                allow_zen_conversion=True,
                structured_conf_permitted=True,
            ),
        )


@dataclass(unsafe_hash=True)
class ConfigRange:
    start: InitVar[int]
    stop: InitVar[int]
    step: InitVar[int]
    _target_: str = field(default=BuildsFn._get_obj_path(range), init=False)
    _args_: tuple[int, ...] = field(default=(), init=False, repr=False)
    CBuildsFn: InitVar[type[BuildsFn[Any]]]

    def __post_init__(
        self, start: int, stop: int, step: int, CBuildsFn: type[BuildsFn[Any]]
    ) -> None:
        del CBuildsFn
        self._args_ = (start, stop, step)


@dataclass(unsafe_hash=True)
class ConfigTimeDelta:
    CBuildsFn: InitVar[type[BuildsFn[Any]]]
    days: float = 0.0
    seconds: float = 0.0
    microseconds: float = 0.0
    milliseconds: float = 0.0
    minutes: float = 0.0
    hours: float = 0.0
    weeks: float = 0.0
    _target_: str = field(default=BuildsFn._get_obj_path(timedelta), init=False)

    def __post_init__(self, CBuildsFn: type[BuildsFn[Any]]) -> None:
        del CBuildsFn


@dataclass(unsafe_hash=True)
class ConfigFromDefaultDict:
    dict_: dict[Any, Any]
    default_factory: Any = field(init=False)
    CBuildsFn: InitVar[type[BuildsFn[Any]]]
    _target_: str = BuildsFn._get_obj_path(as_default_dict)

    def __post_init__(self, CBuildsFn: type[BuildsFn[Any]]) -> None:
        assert isinstance(self.dict_, defaultdict)
        self.default_factory = CBuildsFn.just(self.dict_.default_factory)
        out = CBuildsFn._make_hydra_compatible(
            dict(self.dict_),
            convert_dataclass=True,
            allow_zen_conversion=True,
            structured_conf_permitted=True,
        )
        assert isinstance(out, dict)
        self.dict_ = out


ZEN_VALUE_CONVERSION[defaultdict] = lambda dict_, CBuildsFn: ConfigFromDefaultDict(
    dict_, CBuildsFn
)

ZEN_VALUE_CONVERSION[set] = partial(
    ConfigFromTuple, _target_=BuildsFn._get_obj_path(set)
)
ZEN_VALUE_CONVERSION[frozenset] = partial(
    ConfigFromTuple, _target_=BuildsFn._get_obj_path(frozenset)
)
ZEN_VALUE_CONVERSION[deque] = partial(
    ConfigFromTuple, _target_=BuildsFn._get_obj_path(deque)
)

if bytes in ZEN_SUPPORTED_PRIMITIVES:  # pragma: no cover
    ZEN_VALUE_CONVERSION[bytes] = partial(
        ConfigFromTuple, _target_=BuildsFn._get_obj_path(bytes)
    )

ZEN_VALUE_CONVERSION[bytearray] = partial(
    ConfigFromTuple, _target_=BuildsFn._get_obj_path(bytearray)
)
ZEN_VALUE_CONVERSION[range] = lambda value, CBuildsFn: ConfigRange(
    value.start,
    value.stop,
    value.step,
    CBuildsFn=CBuildsFn,
)
ZEN_VALUE_CONVERSION[timedelta] = lambda value, CBuildsFn: ConfigTimeDelta(
    days=value.days,
    seconds=value.seconds,
    microseconds=value.microseconds,
    CBuildsFn=CBuildsFn,
)
ZEN_VALUE_CONVERSION[Counter] = partial(
    ConfigFromDict, _target_=BuildsFn._get_obj_path(Counter)
)
ZEN_VALUE_CONVERSION[functools.partial] = _unpack_partial
