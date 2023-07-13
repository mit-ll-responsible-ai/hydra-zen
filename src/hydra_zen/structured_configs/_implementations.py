# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
import inspect
import sys
import warnings
from dataclasses import (  # use this for runtime checks
    MISSING,
    Field as _Field,
    dataclass,
    field,
    fields,
    is_dataclass,
    make_dataclass,
)
from enum import Enum
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
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
    get_type_hints,
    overload,
)

from omegaconf import DictConfig, ListConfig
from typing_extensions import Final, Literal, ParamSpec, TypeAlias, dataclass_transform

from hydra_zen._compatibility import (
    HYDRA_SUPPORTED_PRIMITIVES,
    ZEN_SUPPORTED_PRIMITIVES,
)
from hydra_zen.errors import (
    HydraZenDeprecationWarning,
    HydraZenUnsupportedPrimitiveError,
    HydraZenValidationError,
)
from hydra_zen.funcs import get_obj
from hydra_zen.structured_configs import _utils
from hydra_zen.structured_configs._type_guards import safe_getattr
from hydra_zen.typing import (
    Builds,
    DataclassOptions,
    PartialBuilds,
    SupportedPrimitive,
    ZenWrappers,
)
from hydra_zen.typing._implementations import (
    AllConvert,
    AnyBuilds,
    BuildsWithSig,
    DataClass_,
    DefaultsList,
    Field,
    HasTarget,
    InstOrType,
    ZenConvert,
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
    is_just,
    is_old_partial_builds,
    safe_getattr,
    uses_zen_processing,
)

_T = TypeVar("_T")
Importable = TypeVar("Importable", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")
Field_Entry: TypeAlias = Tuple[str, type, Field[Any]]

# default zen_convert settings for `builds` and `hydrated_dataclass`
_BUILDS_CONVERT_SETTINGS = AllConvert(dataclass=True, flat_target=True)

# stores type -> value-conversion-fn
# for types with specialized support from hydra-zen
ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}

# signature param-types
_POSITIONAL_ONLY: Final = inspect.Parameter.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD: Final = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL: Final = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY: Final = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD: Final = inspect.Parameter.VAR_KEYWORD


_builtin_function_or_method_type = type(len)
# fmt: off
_lru_cache_type = type(functools.lru_cache(maxsize=128)(lambda: None))  # pragma: no branch
# fmt: on

_BUILTIN_TYPES: Final = (_builtin_function_or_method_type, _lru_cache_type)

del _lru_cache_type
del _builtin_function_or_method_type


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


def mutable_value(x: _T, *, zen_convert: Optional[ZenConvert] = None) -> _T:
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
        x = sanitize_collection(x, convert_dataclass=settings["dataclass"])
        return field(default_factory=lambda: cast(x))
    return field(default_factory=lambda: x)


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
) -> Callable[[Type[_T]], Type[_T]]:
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
            }
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
            kwargs: Dict[str, Any] = {}

        out = builds(
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

        return out

    return wrapper


@dataclass(unsafe_hash=True)
class Just:
    """Just[T] is a config that returns T when instantiated."""

    path: str
    _target_: str = field(default="hydra_zen.funcs.get_obj", init=False, repr=False)


def _just(obj: Any) -> Just:
    return Just(path=_utils.get_obj_path(obj))


def _is_ufunc(value: Any) -> bool:
    # checks without importing numpy
    numpy = sys.modules.get("numpy")
    if numpy is None:  # pragma: no cover
        # we do actually cover this branch some runs of our CI,
        # but our coverage job installs numpy
        return False
    return isinstance(value, numpy.ufunc)


def _check_instance(*target_types: str, value: "Any", module: str):  # pragma: no cover
    """Checks if value is an instance of any of the target types imported
    from the specified module.

    Returns `False` if module/target type doesn't exists (e.g. not installed).
    This is useful for gracefully handling specialized logic for optional dependencies.
    """
    mod = sys.modules.get(module)
    if mod is None:
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


_is_numpy_array_func_dispatcher = functools.partial(
    _check_instance, "_ArrayFunctionDispatcher", module="numpy.core._multiarray_umath"
)

_is_jax_compiled_func = functools.partial(
    _check_instance, "CompiledFunction", "PjitFunction", module="jaxlib.xla_extension"
)

_is_jax_unspecified = functools.partial(
    _check_instance, "UnspecifiedValue", module="jax._src.interpreters.pxla"
)

_is_torch_optim_required = functools.partial(
    _check_instance, "required", module="torch.optim.optimizer"
)


def _check_for_dynamically_defined_dataclass_type(target_path: str, value: Any) -> None:
    if target_path.startswith("types."):
        raise HydraZenUnsupportedPrimitiveError(
            f"Configuring {value}: Cannot auto-config an instance of a "
            f"dynamically-generated dataclass type (e.g. one created from "
            f"`hydra_zen.make_config` or `dataclasses.make_dataclass`). "
            f"Consider disabling auto-config support for dataclasses here."
        )


def sanitized_default_value(
    value: Any,
    allow_zen_conversion: bool = True,
    *,
    error_prefix: str = "",
    field_name: str = "",
    structured_conf_permitted: bool = True,
    convert_dataclass: bool,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    zen_dataclass: Optional[DataclassOptions] = None,
) -> Any:
    """Converts `value` to Hydra-supported type if necessary and possible. Otherwise
    raises `HydraZenUnsupportedPrimitiveError`"""
    # Common primitives supported by Hydra.
    # We check exhaustively for all Hydra-supported primitives below but seek to
    # speedup checks for common types here.
    if value is None or type(value) in {str, int, bool, float}:
        return value

    # non-str collection
    if hasattr(value, "__iter__"):
        value = sanitize_collection(
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
                converted_fields[_field.name] = sanitized_default_value(
                    _val,
                    allow_zen_conversion=allow_zen_conversion,
                    field_name=_field.name,
                    convert_dataclass=convert_dataclass,
                )

        out = builds(
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
                and inspect.isclass(value)
            )
            or inspect.ismethod(value)
            or isinstance(value, _BUILTIN_TYPES)
            or _is_ufunc(value)
            or _is_numpy_array_func_dispatcher(value=value)
            or _is_jax_compiled_func(value=value)
        )
    ):
        # `value` is importable callable -- create config that will import
        # `value` upon instantiation
        out = _just(value)
        if convert_dataclass and is_dataclass(value):
            _check_for_dynamically_defined_dataclass_type(
                safe_getattr(out, JUST_FIELD_NAME), value
            )
        return out

    resolved_value = value
    type_of_value = type(resolved_value)

    # hydra-zen supported primitives from stdlib
    #
    # Note: we don't use isinstance because we don't permit subclasses of supported
    # primitives
    if allow_zen_conversion and type_of_value in ZEN_SUPPORTED_PRIMITIVES:
        type_ = type(resolved_value)
        conversion_fn = ZEN_VALUE_CONVERSION[type_]

        resolved_value = conversion_fn(resolved_value)
        type_of_value = type(resolved_value)

    if type_of_value in HYDRA_SUPPORTED_PRIMITIVES or (
        structured_conf_permitted
        and (
            is_dataclass(resolved_value)
            or isinstance(resolved_value, (Enum, ListConfig, DictConfig))
        )
    ):
        return resolved_value

    # pydantic objects
    pydantic = sys.modules.get("pydantic")
    if pydantic is not None and isinstance(value, pydantic.fields.FieldInfo):
        _val = (
            value.default_factory()  # type: ignore
            if value.default_factory is not None  # type: ignore
            else value.default  # type: ignore
        )
        if isinstance(_val, pydantic.fields.UndefinedType):
            return MISSING

        return sanitized_default_value(
            _val,
            allow_zen_conversion=allow_zen_conversion,
            error_prefix=error_prefix,
            field_name=field_name,
            structured_conf_permitted=structured_conf_permitted,
            convert_dataclass=convert_dataclass,
            hydra_convert=hydra_convert,
            hydra_recursive=hydra_recursive,
        )

    if isinstance(value, str):
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


def sanitize_collection(
    x: _T,
    *,
    convert_dataclass: bool,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
) -> _T:
    """Pass contents of lists, tuples, or dicts through sanitized_default_values"""
    type_x = type(x)
    if type_x in {list, tuple}:
        return type_x(sanitized_default_value(_x, convert_dataclass=convert_dataclass, hydra_convert=hydra_convert, hydra_recursive=hydra_recursive) for _x in x)  # type: ignore
    elif type_x is dict:
        return {
            # Hydra doesn't permit structured configs for keys, thus we only
            # support its basic primitives here.
            sanitized_default_value(
                k,
                allow_zen_conversion=False,
                structured_conf_permitted=False,
                error_prefix="Configuring dictionary key:",
                convert_dataclass=False,
            ): sanitized_default_value(
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


def sanitized_field(
    value: Any,
    init: bool = True,
    allow_zen_conversion: bool = True,
    *,
    error_prefix: str = "",
    field_name: str = "",
    convert_dataclass: bool,
) -> Field[Any]:
    value = sanitized_default_value(
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
        is_dataclass(value) and not isinstance(value, type) and value.__hash__ is None
    ):
        return cast(
            Field[Any],
            mutable_value(value, zen_convert={"dataclass": convert_dataclass}),
        )
    return _utils.field(default=value, init=init)


def _get_sig_obj(target):
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
def builds(
    __hydra_target: Type[BuildsWithSig[Type[R], P]],
    *,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: Literal[True],
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[()] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> Type[BuildsWithSig[Type[R], P]]:
    ...


@overload
def builds(
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
    builds_bases: Tuple[()] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> Type[BuildsWithSig[Type[R], P]]:
    ...


# partial=False, pop-sig=bool
@overload
def builds(
    __hydra_target: Type[AnyBuilds[Importable]],
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:
    ...


# partial=False, pop-sig=bool
@overload
def builds(
    __hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:
    ...


# partial=True, pop-sig=bool
@overload
def builds(
    __hydra_target: Type[AnyBuilds[Importable]],
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True] = ...,
    populate_full_signature: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:
    ...


# partial=True, pop-sig=bool
@overload
def builds(
    __hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True] = ...,
    populate_full_signature: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:
    ...


# partial=bool, pop-sig=False
@overload
def builds(
    __hydra_target: Type[AnyBuilds[Importable]],
    *pos_args: SupportedPrimitive,
    zen_partial: Optional[bool] = ...,
    populate_full_signature: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
    ...


# partial=bool, pop-sig=False
@overload
def builds(
    __hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Optional[bool] = ...,
    populate_full_signature: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
    ...


# partial=bool, pop-sig=bool
@overload
def builds(
    __hydra_target: Union[Callable[P, R], Type[Builds[Importable]], Importable],
    *pos_args: SupportedPrimitive,
    zen_partial: Optional[bool],
    populate_full_signature: bool = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    hydra_defaults: Optional[DefaultsList] = ...,
    dataclass_name: Optional[str] = ...,
    builds_bases: Tuple[Type[DataClass_], ...] = ...,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Type[BuildsWithSig[Type[R], P]],
]:
    ...


def builds(
    *pos_args: Union[
        Importable, Callable[P, R], Type[AnyBuilds[Importable]], SupportedPrimitive
    ],
    zen_partial: Optional[bool] = None,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = False,
    zen_convert: Optional[ZenConvert] = None,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    hydra_defaults: Optional[DefaultsList] = None,
    builds_bases: Tuple[Type[DataClass_], ...] = (),
    zen_dataclass: Optional[DataclassOptions] = None,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]],
    Type[PartialBuilds[Importable]],
    Type[BuildsWithSig[Type[R], P]],
]:
    """builds(hydra_target, /, *pos_args, zen_partial=None, zen_wrappers=(), zen_meta=None, populate_full_signature=False, hydra_recursive=None, hydra_convert=None, hydra_defaults=None, frozen=False, dataclass_name=None, builds_bases=(), **kwargs_for_target)

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
        If ``True``, then the resulting config's signature and fields will be populated
        according to the signature of ``<hydra_target>``; values also specified in
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

        Additionally, the `module` field can be specified to enable pickle
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
        The provided configured value cannot be serialized by Hydra, nor does hydra-zen
        provide specialized support for it. See :ref:`valid-types` for more details.

    Notes
    -----
    The following pseudo code conveys the core functionality of `builds`:

    .. code-block:: python

       from dataclasses import make_dataclass

       def builds(target, populate_full_signature=False, **kw):
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

    The following config will have a signature that matches ``f``; the
    annotations and default values of the parameters of ``f`` are explicitly
    incorporated into the config.

    >>> # signature: `Builds_bar(x: bool, y: str = 'foo')`
    >>> Conf = builds(bar, populate_full_signature=True)
    >>> pyaml(Conf)
    _target_: __main__.bar
    x: ???
    'y': foo

    Annotations will be used by Hydra to provide limited runtime type-checking during
    instantiation. Here, we'll pass a float for ``x``, which expects a boolean value.

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
    >>> issubclass(ChildConf, ParentConf)
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
    zen_convert_settings = _utils.merge_settings(zen_convert, _BUILDS_CONVERT_SETTINGS)
    if zen_dataclass is None:
        zen_dataclass = {}

    # initial validation
    _utils.parse_dataclass_options(zen_dataclass)

    if "frozen" in kwargs_for_target:
        warnings.warn(
            HydraZenDeprecationWarning(
                "Specifying `builds(..., frozen=<...>)` is deprecated. Instead, "
                "specify `builds(..., zen_dataclass={'frozen': <...>})"
            ),
            stacklevel=2,
        )
        zen_dataclass["frozen"] = kwargs_for_target.pop("frozen")  # type: ignore

    if "dataclass_name" in kwargs_for_target:
        warnings.warn(
            HydraZenDeprecationWarning(
                "Specifying `builds(..., dataclass_name=<...>)` is deprecated. "
                "Instead specify `builds(..., zen_dataclass={'cls_name': <...>})"
            ),
            stacklevel=2,
        )
        zen_dataclass["cls_name"] = kwargs_for_target.pop("dataclass_name")  # type: ignore
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

    if not callable(target):
        raise TypeError(
            BUILDS_ERROR_PREFIX
            + "In `builds(<target>, ...), `<target>` must be callable/instantiable"
        )

    if not isinstance(populate_full_signature, bool):
        raise TypeError(
            f"`populate_full_signature` must be a boolean type, got: {populate_full_signature}"
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
    if (
        zen_convert_settings["flat_target"]
        and isinstance(target, type)
        and is_dataclass(target)
        and hasattr(target, TARGET_FIELD_NAME)
    ):
        # pass through _target_ field
        target_path = safe_getattr(target, TARGET_FIELD_NAME)
    else:
        target_path = _utils.get_obj_path(target)

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
                wrapper = ZEN_VALUE_CONVERSION[functools.partial](wrapper)

            if is_builds(wrapper):
                # If Hydra's locate function starts supporting importing literals
                # – or if we decide to ship our own locate function –
                # then we should get the target of `wrapper` and make sure it is callable
                if is_just(wrapper):
                    # `zen_wrappers` handles importing string; we can
                    # eliminate the indirection of Just and "flatten" this
                    # config
                    validated_wrappers.append(safe_getattr(wrapper, JUST_FIELD_NAME))
                else:
                    if hydra_recursive is False:
                        warnings.warn(
                            "A structured config was supplied for `zen_wrappers`. Its parent config has "
                            "`hydra_recursive=False`.\n If this value is not toggled to `True`, the config's "
                            "instantiation will result in an error"
                        )
                    validated_wrappers.append(wrapper)

            elif callable(wrapper):
                validated_wrappers.append(_utils.get_obj_path(wrapper))

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

    target_field: List[Union[Tuple[str, Type[Any]], Tuple[str, Type[Any], Any]]]

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
    base_hydra_partial: Optional[bool] = None  # state of closest parent with _partial_
    base_zen_partial: Optional[bool] = None  # state of closest parent with _zen_partial

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
                _utils.field(default=target_path, init=False),
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
                    Tuple[str, ...],
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
                            Union[str, Builds[Any]], Tuple[Union[str, Builds[Any]], ...]
                        ],
                        _utils.field(default=validated_wrappers[0], init=False),
                    ),
                )
            else:
                target_field.append(
                    (
                        ZEN_WRAPPERS_FIELD_NAME,
                        Union[
                            Union[str, Builds[Any]], Tuple[Union[str, Builds[Any]], ...]
                        ],
                        _utils.field(default=validated_wrappers, init=False),
                    ),
                )
    else:
        target_field = [
            (
                TARGET_FIELD_NAME,
                str,
                _utils.field(default=target_path, init=False),
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
            (CONVERT_FIELD_NAME, str, _utils.field(default=hydra_convert, init=False))
        )

    if hydra_defaults is not None:
        if not _utils.valid_defaults_list(hydra_defaults):
            raise HydraZenValidationError(
                f"`hydra_defaults` must be type `None | list[str | dict[str, str | list[str] | None ]]`"
                f", Got: {repr(hydra_defaults)}"
            )
        hydra_defaults = sanitize_collection(hydra_defaults, convert_dataclass=False)
        base_fields.append(
            (
                DEFAULTS_LIST_FIELD_NAME,
                List[Any],
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
                Tuple[Any, ...],
                _utils.field(
                    default=tuple(
                        sanitized_default_value(
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

    _sig_target = _get_sig_obj(target)

    try:
        # We want to rely on `inspect.signature` logic for raising
        # against an uninspectable sig, before we start inspecting
        # class-specific attributes below.
        signature_params = dict(inspect.signature(target).parameters)
    except ValueError:
        if populate_full_signature:
            raise ValueError(
                BUILDS_ERROR_PREFIX
                + f"{target} does not have an inspectable signature. "
                f"`builds({_utils.safe_name(target)}, populate_full_signature=True)` is not supported"
            )
        signature_params: Dict[str, inspect.Parameter] = {}
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

        if _sig_target is not target:
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

    if is_dataclass(target):
        # Update `signature_params` so that any param with `default=<factory>`
        # has its default replaced with `<factory>()`
        # If this is a mutable value, `builds` will automatically re-pack
        # it using a default factory
        _fields = {f.name: f for f in fields(target)}
        _update = {}
        for name, param in signature_params.items():
            if name not in _fields:
                # field is InitVar
                continue
            f = _fields[name]
            if f.default_factory is not MISSING:
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
        type_hints: Dict[str, Any] = {}

    sig_by_kind: Dict[Any, List[inspect.Parameter]] = {
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
    nameable_params_in_sig: Set[str] = {
        p.name
        for p in chain(sig_by_kind[_POSITIONAL_OR_KEYWORD], sig_by_kind[_KEYWORD_ONLY])
    }

    if not _pos_args and builds_bases:
        # pos_args is potentially inherited
        for _base in builds_bases:
            _pos_args = safe_getattr(_base, POS_ARG_FIELD_NAME, ())

            # validates
            _pos_args = tuple(
                sanitized_default_value(
                    x, allow_zen_conversion=False, convert_dataclass=False
                )
                for x in _pos_args
            )
            if _pos_args:
                break

    fields_set_by_bases: Set[str] = {
        _field.name
        for _base in builds_bases
        for _field in fields(_base)
        if _field.name not in HYDRA_FIELD_NAMES and not _field.name.startswith("_zen_")
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
    user_specified_named_params: Dict[str, Tuple[str, type, Any]] = {
        name: (name, type_hints.get(name, Any), value)
        for name, value in kwargs_for_target.items()
    }

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
        _fields_with_default_values: List[Field_Entry] = []

        # we need to keep track of what user-specified params we have set
        _seen: Set[str] = set()

        for n, param in enumerate(signature_params.values()):
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
                sanitized_default_value(
                    field_.default,
                    allow_zen_conversion=False,
                    error_prefix=BUILDS_ERROR_PREFIX,
                    field_name=field_.name + " (set via inheritance)",
                    convert_dataclass=False,
                )
            del field_

    # sanitize all types and configured values
    sanitized_base_fields: List[
        Union[Tuple[str, Any], Tuple[str, Any, Field[Any]]]
    ] = []

    for item in base_fields:
        name = item[0]
        type_ = item[1]
        if len(item) == 2:
            sanitized_base_fields.append((name, _utils.sanitized_type(type_)))
        else:
            assert len(item) == 3, item
            value = item[-1]

            if not isinstance(value, _Field):
                _field = sanitized_field(
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
                _utils.sanitized_type(type_, wrap_optional=sanitized_value is None)
                if _retain_type_info(
                    type_=type_, value=sanitized_value, hydra_recursive=hydra_recursive
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
                f"\n\nThe docstring for {_utils.safe_name(target)} :\n\n" + target_doc
            )

    assert requires_zen_processing is uses_zen_processing(out)

    # _partial_=True should never be relied on when zen-processing is being used.
    assert not (
        requires_zen_processing and safe_getattr(out, PARTIAL_FIELD_NAME, False)
    )

    return cast(Union[Type[Builds[Importable]], Type[BuildsWithSig[Type[R], P]]], out)


@overload
def get_target(obj: InstOrType[Builds[_T]]) -> _T:
    ...


@overload
def get_target(obj: HasTarget) -> Any:
    ...


def get_target(obj: HasTarget) -> Any:
    """
    Returns the target-object from a targeted config.

    Parameters
    ----------
    obj : HasTarget
        An object with a ``_target_`` attribute.

    Returns
    -------
    target : Any

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

    >>> get_target(A)
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

    if isinstance(target, str):
        target = get_obj(path=target)
    else:
        # Hydra 1.1.0 permits objects-as-_target_ instead of strings
        # https://github.com/facebookresearch/hydra/issues/1017
        pass  # makes sure we cover this branch in tests

    return target
