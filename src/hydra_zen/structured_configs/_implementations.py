# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
from collections import defaultdict
from dataclasses import Field, dataclass, field, fields, is_dataclass, make_dataclass
from functools import partial
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    overload,
)

from typing_extensions import Final, Literal

from hydra_zen.funcs import get_obj, partial
from hydra_zen.structured_configs import _utils
from hydra_zen.typing import Builds, Importable, Just, PartialBuilds

try:
    # used to check if default values are ufuncs
    from numpy import ufunc
except ImportError:  # pragma: no cover
    ufunc = None

__all__ = ["builds", "just", "hydrated_dataclass", "mutable_value"]

_T = TypeVar("_T")

_TARGET_FIELD_NAME: Final[str] = "_target_"
_RECURSIVE_FIELD_NAME: Final[str] = "_recursive_"
_CONVERT_FIELD_NAME: Final[str] = "_convert_"
_PARTIAL_TARGET_FIELD_NAME: Final[str] = "_partial_target_"
_POS_ARG_FIELD_NAME: Final[str] = "_args_"
_HYDRA_FIELD_NAMES: FrozenSet[str] = frozenset(
    (
        _TARGET_FIELD_NAME,
        _RECURSIVE_FIELD_NAME,
        _CONVERT_FIELD_NAME,
        _PARTIAL_TARGET_FIELD_NAME,
        _POS_ARG_FIELD_NAME,
    )
)

_POSITIONAL_ONLY: Final = inspect.Parameter.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD: Final = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL: Final = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY: Final = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD: Final = inspect.Parameter.VAR_KEYWORD

_builtin_function_or_method_type = type(len)


def mutable_value(x: Any) -> Field:
    """Used to set a mutable object as a default value for a field
    in a dataclass.

    This is an alias for `field(default_factory=lambda: x)`

    Examples
    --------
    >>> from hydra_zen import mutable_value
    >>> from dataclasses import dataclass

    See https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    >>> @dataclass
    ... class HasMutableDefault
    ...     a_list: list  = [1, 2, 3]  # error: mutable default

    Using `mutable_value` to specify the default list:

    >>> @dataclass
    ... class HasMutableDefault
    ...     a_list: list  = mutable_value([1, 2, 3])  # ok"""
    cast = type(x)  # ensure that we return a copy of the default value
    return field(default_factory=lambda: cast(x))


Field_Entry = Tuple[str, type, Field]


# Alternate form, from PEP proposal:
# https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md
#
# This enables static checkers to work with third-party decorators that create
# dataclass-like objects
def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]:
    # If used within a stub file, the following implementation can be
    # replaced with "...".
    return lambda a: a


@__dataclass_transform__()
def hydrated_dataclass(
    target: Callable,
    *pos_args: Any,
    populate_full_signature: bool = False,
    hydra_partial: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
) -> Callable[[Type[_T]], Type[_T]]:
    """A decorator that uses `hydra_zen.builds` to create a dataclass with the appropriate
    hydra-specific fields for specifying a structured config [1]_.

    Parameters
    ----------
    target : Union[Instantiable, Callable]
        The object to be instantiated/called.

    *pos_args: Any
        Positional arguments passed to `target`.

        Arguments specified positionally are not included in the dataclass' signature and
        are stored as a tuple bound to in the ``_args_`` field.

    populate_full_signature : bool, optional (default=False)
        If True, then the resulting dataclass's __init__ signature and fields
        will be populated according to the signature of `target`.

        Values specified in **kwargs_for_target take precedent over the corresponding
        default values from the signature.

    hydra_partial : Optional[bool] (default=False)
        If True, then hydra-instantiation produces `functools.partial(target, **kwargs)`

    hydra_recursive : bool, optional (default=True)
        If True, then upon hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting dataclass.

    hydra_convert: Optional[Literal["none", "partial", "all"]] (default="none")
        Determines how hydra handles the non-primitive objects passed to `target` [3]_.

        - `"none"`: Passed objects are DictConfig and ListConfig, default
        - `"partial"`: Passed objects are converted to dict and list, with
          the exception of Structured Configs (and their fields).
        - `"all"`: Passed objects are dicts, lists and primitives without
          a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting dataclass.

    frozen : bool, optional (default=False)
        If `True`, the resulting dataclass will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise `FrozenInstanceError`
        at runtime.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies

    Examples
    --------
    A simple usage of `hydrated_dataclass`. Here, we specify a structured config

    >>> from hydra_zen import hydrated_dataclass, instantiate
    >>> @hydrated_dataclass(target=dict)
    ... class DictConf:
    ...     x : int = 2
    ...     y : str = 'hello'

    >>> instantiate(DictConf(x=10))  # override default `x`
    {'x': 10, 'y': 'hello'}

    We can also design a configuration that only partially instantiates our target.

    >>> def power(x: float, exponent: float) -> float: return x ** exponent
    >>> @hydrated_dataclass(target=power, hydra_partial=True)
    ... class PowerConf:
    ...     exponent : float = 2.0

    >>> partiald_power = instantiate(PowerConf)
    >>> partiald_power(10.0)
    100.0

    Inheritance can be used to compose configurations

    >>> from dataclasses import dataclass
    >>> from torch.optim import AdamW
    >>> @dataclass
    ... class AdamBaseConfig:
    ...     lr : float = 0.001
    ...     eps : float = 1e-8

    >>> @hydrated_dataclass(target=AdamW, hydra_partial=True)
    ... class AdamWConfig(AdamBaseConfig):
    ...     weight_decay : float = 0.01
    >>> instantiate(AdamWConfig)
    functools.partial(<class 'torch.optim.adamw.AdamW'>, lr=0.001, eps=1e-08, weight_decay=0.01)

    Because this decorator uses `hyda_utils.builds` under the hood, common mistakes like misspelled
    parameters will be caught upon constructing the structured config.

    >>> @hydrated_dataclass(target=AdamW, hydra_partial=True)
    ... class AdamWConfig(AdamBaseConfig):
    ...     wieght_decay : float = 0.01  # i before e, right!?
    TypeError: Building: AdamW ..
    The following unexpected keyword argument(s) for torch.optim.adamw.AdamW was specified via inheritance
    from a base class: wieght_decay
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
        decorated_obj = dataclass(frozen=frozen)(decorated_obj)

        return builds(
            target,
            *pos_args,
            populate_full_signature=populate_full_signature,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            hydra_partial=hydra_partial,
            builds_bases=(decorated_obj,),
            dataclass_name=decorated_obj.__name__,
            frozen=frozen,
        )

    return wrapper


def just(obj: Importable) -> Type[Just[Importable]]:
    """Produces a structured config that, when instantiated by hydra, 'just'
    returns `obj`.

    This is convenient for specifying a particular, un-instantiated object as part of your
    configuration.

    Parameters
    ----------
    obj : Importable
        The object that will be instantiated from this config.

    Returns
    -------
    types.JustObj
        The dataclass object that is designed as a structured config.

    Examples
    --------
    >>> from hydra_zen import just, instantiate, to_yaml
    >>> just_range = just(range)
    >>> range is instantiate(just_range)
    True
    >>> just_range._target_
    'hydra_zen.funcs.identity'
    >>> just_range.path
    'builtins.range'
    """
    try:
        obj_path = _utils.get_obj_path(obj)
    except AttributeError:
        raise AttributeError(
            f"`just({obj})`: `obj` is not importable; it is missing the attributes `__module__` and/or `__qualname__`"
        )

    out_class = make_dataclass(
        ("Just_" + _utils.safe_name(obj)),
        [
            (
                _TARGET_FIELD_NAME,
                str,
                field(default=_utils.get_obj_path(get_obj), init=False),
            ),
            (
                "path",
                str,
                field(
                    default=obj_path,
                    init=False,
                ),
            ),
        ],
    )
    out_class.__doc__ = (
        f"A structured config designed to return {obj} when it is instantiated by hydra"
    )

    return out_class


def create_just_if_needed(value: _T) -> Union[_T, Type[Just[_T]]]:
    # Hydra can serialize dataclasses directly, thus we
    # don't want to wrap these in `just`

    if callable(value) and (
        inspect.isfunction(value)
        or (inspect.isclass(value) and not is_dataclass(value))
        or isinstance(value, _builtin_function_or_method_type)
        or (ufunc is not None and isinstance(value, ufunc))
    ):
        return just(value)

    return value


def sanitized_default_value(value: Any) -> Union[Field, Type[Just]]:
    if isinstance(value, _utils.KNOWN_MUTABLE_TYPES):
        return mutable_value(value)
    resolved_value = create_just_if_needed(value)
    return field(default=value) if value is resolved_value else resolved_value


# overloads when `hydra_partial=False`
@overload
def builds(
    target: Importable,
    *pos_args: Any,
    populate_full_signature: bool = False,
    hydra_partial: Literal[False] = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    frozen: bool = False,
    **kwargs_for_target,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


# overloads when `hydra_partial=True`
@overload
def builds(
    target: Importable,
    *pos_args: Any,
    populate_full_signature: bool = False,
    hydra_partial: Literal[True],
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    frozen: bool = False,
    **kwargs_for_target,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


# overloads when `hydra_partial: bool`
@overload
def builds(
    target: Importable,
    *pos_args: Any,
    populate_full_signature: bool = False,
    hydra_partial: bool,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    frozen: bool = False,
    **kwargs_for_target,
) -> Union[
    Type[Builds[Importable]], Type[PartialBuilds[Importable]]
]:  # pragma: no cover
    ...


def builds(
    target: Importable,
    *pos_args: Any,
    populate_full_signature: bool = False,
    hydra_partial: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    **kwargs_for_target,
) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
    """Returns a dataclass object that configures `target` with user-specified and auto-populated parameter values.

    The resulting dataclass is specifically a structured config [1]_ that enables Hydra to initialize/call
    `target` either fully or partially. See Notes for additional features and explanation of implementation details.

    Parameters
    ----------
    target : Union[Instantiable, Callable]
        The object to be instantiated/called

    *pos_args: Any
        Positional arguments passed to `target`.

        Arguments specified positionally are not included in the dataclass' signature and
        are stored as a tuple bound to in the ``_args_`` field.

    **kwargs_for_target : Any
        The keyword arguments passed to `target(...)`.

        The arguments specified here solely determine the fields and init-parameters of the
        resulting dataclass, unless `populate_full_signature=True` is specified (see below).

    populate_full_signature : bool, optional (default=False)
        If `True`, then the resulting dataclass's signature and fields will be populated
        according to the signature of `target`.

        Values specified in **kwargs_for_target take precedent over the corresponding
        default values from the signature.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    hydra_partial : bool, optional (default=False)
        If True, then hydra-instantiation produces `functools.partial(target, **kwargs_for_target)`,
        this enables the partial-configuration of objects.

        Specifying `hydra_partial=True` and `populate_full_signature=True` together will
        populate the dataclass' signature only with parameters that are specified by the
        user or that have default values specified in the target's signature. I.e. it is
        presumed that un-specified parameters are to be excluded from the partial configuration.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then upon hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting dataclass.

    hydra_convert: Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how hydra handles the non-primitive objects passed to `target` [3]_.

        - `"none"`: Passed objects are DictConfig and ListConfig, default
        - `"partial"`: Passed objects are converted to dict and list, with
          the exception of Structured Configs (and their fields).
        - `"all"`: Passed objects are dicts, lists and primitives without
          a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting dataclass.

    frozen : bool, optional (default=False)
        If `True`, the resulting dataclass will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise `FrozenInstanceError`
        at runtime.

    builds_bases : Tuple[DataClass, ...]
        Specifies a tuple of parent classes that the resulting dataclass inherits from.
        A `PartialBuilds` class (resulting from `hydra_partial=True`) cannot be a parent
        of a `Builds` class (i.e. where `hydra_partial=False` was specified).

    dataclass_name : Optional[str]
        If specified, determines the name of the returned class object.

    Returns
    -------
    builder : Builds[target]
        The structured config (a dataclass with the field: _target_ populated).

    Raises
    ------
    TypeError
        One or more unexpected arguments were specified via **kwargs_for_target, which
        are not compatible with the signature of `target`.

    Notes
    -----
    Type annotations are inferred from the target's signature and are only retained if they are compatible
    with hydra's limited set of supported annotations; otherwise `Any` is specified.

    `builds` provides runtime validation of user-specified named arguments against the target's signature.
    This helps to ensure that typos in field names fail early and explicitly.

    Mutable values are automatically specified using a default factory [4]_.

    `builds(...)` is annotated to return the generic protocols `Builds` and `PartialBuilds`, which are
    available in `hydra_zen.typing`. These are leveraged by `hydra_zen.instantiate` to provide static
    analysis tooling with enhanced context.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [4] https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    Examples
    --------
    Basic Usage:

    >>> from hydra_zen import builds, instantiate
    >>> builds(dict, a=1, b='x')  # makes a dataclass that will "build" a dictionary with the specified fields
    types.Builds_dict
    >>> instantiate(builds(dict, a=1, b='x'))  # using hydra to build the dictionary
    {'a': 1, 'b': 'x'}

    >>> Conf = builds(len, [1, 2, 3])  # specifying positional arguments
    >>> Conf._args_
    ([1, 2, 3],)
    >>> instantiate(Conf)
    3

    Using `builds` with partial instantiation

    >>> def a_two_tuple(x: int, y: float): return x, y
    >>> PartialBuildsFunc = builds(a_two_tuple, x=1, hydra_partial=True)  # specifies only `x`
    >>> partial_func = instantiate(PartialBuildsFunc)
    >>> partial_func
    functools.partial(<function a_two_tuple at 0x00000220A7820EE0>, x=1)
    >>> partial_func(y=22)
    (1, 22)

    Auto-populating parameters:

    >>> # signature: `Builds_a_two_tuple(x: int, y: float)`
    >>> Conf = builds(a_two_tuple, populate_full_signature)
    >>> instantiate(Conf(x=1, y=10.0))
    (1, 10.0)

    Inheritance:

    >>> ParentConf = builds(dict, a=1, b=2)
    >>> ChildConf = builds(dict, b=-2, c=-3, builds_bases=(ParentConf,))
    >>> instantiate(ChildConf)
    {'a': 1, 'b': -2, 'c': -3}
    >>> issubclass(ChildConf, ParentConf)
    True
    """

    if not callable(target):
        raise TypeError(
            _utils.building_error_prefix(target)
            + "In `builds(target, ...), `target` must be callable/instantiable"
        )

    if not isinstance(populate_full_signature, bool):
        raise TypeError(
            f"`populate_full_signature` must be a boolean type, got: {populate_full_signature}"
        )

    if hydra_recursive is not None and not isinstance(hydra_recursive, bool):
        raise TypeError(
            f"`hydra_recursive` must be a boolean type, got {hydra_recursive}"
        )

    if not isinstance(hydra_partial, bool):
        raise TypeError(f"`hydra_partial` must be a boolean type, got: {hydra_partial}")

    if hydra_convert is not None and hydra_convert not in {"none", "partial", "all"}:
        raise ValueError(
            f"`hydra_convert` must be 'none', 'partial', or 'all', got: {hydra_convert}"
        )

    if dataclass_name is not None and not isinstance(dataclass_name, str):
        raise TypeError(
            f"`dataclass_name` must be a string or None, got: {dataclass_name}"
        )

    if any(not (is_dataclass(_b) and isinstance(_b, type)) for _b in builds_bases):
        raise TypeError("All `build_bases` must be a tuple of dataclass types")

    if hydra_partial is True and hydra_recursive is False:
        raise ValueError(
            _utils.building_error_prefix(target)
            + "`builds(..., hydra_partial=True)` requires that `hydra_recursive=True`"
        )

    if hydra_partial is True:
        target_field = [
            (
                _TARGET_FIELD_NAME,
                str,
                field(default=_utils.get_obj_path(partial), init=False),
            ),
            (_PARTIAL_TARGET_FIELD_NAME, Any, field(default=just(target), init=False)),
        ]
    else:
        target_field = [
            (
                _TARGET_FIELD_NAME,
                str,
                field(default=_utils.get_obj_path(target), init=False),
            )
        ]

    base_fields: List[Tuple[str, type, Field_Entry]] = target_field

    if hydra_recursive is not None:
        base_fields.append(
            (_RECURSIVE_FIELD_NAME, bool, field(default=hydra_recursive, init=False))
        )

    if hydra_convert is not None:
        base_fields.append(
            (_CONVERT_FIELD_NAME, str, field(default=hydra_convert, init=False))
        )

    if pos_args:
        base_fields.append(
            (
                _POS_ARG_FIELD_NAME,
                Tuple[Any, ...],
                field(
                    default=tuple(create_just_if_needed(x) for x in pos_args),
                    init=False,
                ),
            )
        )

    try:
        signature_params = inspect.signature(target).parameters
        target_has_valid_signature: bool = True
    except ValueError:
        if populate_full_signature:
            raise ValueError(
                _utils.building_error_prefix(target)
                + f"{target} does not have an inspectable signature. "
                f"`builds({target.__name__}, populate_full_signature=True)` is not supported"
            )
        signature_params: Mapping[str, inspect.Parameter] = {}
        # We will turn off signature validation for objects that didn't have
        # a valid signature. This will enable us to do things like `build(dict, a=1)`
        target_has_valid_signature: bool = False

    # this properly resolves forward references, whereas the annotations
    # from signature do not
    try:
        if inspect.isclass(target) and hasattr(type, "__init__"):
            # target is class object...
            # calling `get_type_hints(target)` returns empty dict
            type_hints = get_type_hints(target.__init__)
        else:
            type_hints = get_type_hints(target)
    except (TypeError, NameError):
        # TypeError: Covers case for ufuncs, which do not have inspectable type hints
        # NameError: Covers case for unresolved forward reference
        type_hints = defaultdict(lambda: Any)

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
    nameable_params_in_sig: Set[str] = set(
        p.name
        for p in chain(sig_by_kind[_POSITIONAL_OR_KEYWORD], sig_by_kind[_KEYWORD_ONLY])
    )

    if not pos_args and builds_bases:
        # pos_args is potentially inherited
        for _base in builds_bases:
            pos_args = getattr(_base, _POS_ARG_FIELD_NAME, ())
            if pos_args:
                break

    fields_set_by_bases: Set[str] = {
        _field.name
        for _base in builds_bases
        for _field in fields(_base)
        if _field.name not in _HYDRA_FIELD_NAMES
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
                    _utils.building_error_prefix(target)
                    + f"The following unexpected keyword argument(s) was specified for {_utils.get_obj_path(target)} "
                    f"via `builds`: {', '.join(_unexpected)}"
                )
            if not fields_set_by_bases <= nameable_params_in_sig:
                _unexpected = fields_set_by_bases - nameable_params_in_sig
                raise TypeError(
                    _utils.building_error_prefix(target)
                    + f"The following unexpected keyword argument(s) for {_utils.get_obj_path(target)} "
                    f"was specified via inheritance from a base class: "
                    f"{', '.join(_unexpected)}"
                )

        if pos_args:
            named_args = set(kwargs_for_target).union(fields_set_by_bases)

            # indicates that number of parameters that could be specified by name,
            # but are specified by position
            _num_nameable_args_by_position = max(
                0, len(pos_args) - len(sig_by_kind[_POSITIONAL_ONLY])
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
                            _utils.building_error_prefix(target)
                            + f"Multiple values for argument {param.name} were specified for "
                            f"{_utils.get_obj_path(target)} via `builds`"
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
                    _utils.building_error_prefix(target)
                    + f"{_utils.get_obj_path(target)} takes {_permissible} positional args, but "
                    f"{len(pos_args)} were specified via `builds`"
                )

    # Create valid dataclass fields from the user-specified values
    #
    # user_specified_params: arg-name -> (arg-name, arg-type, field-w-value)
    #  - arg-type: taken from the parameter's annotation in the target's signature
    #    and is resolved to one of the type-annotations supported by hydra if possible,
    #    otherwise, is Any
    #  - arg-value: mutable values are automatically specified using default-factory
    user_specified_named_params: Dict[str, Tuple[str, type, Field]] = {
        name: (
            name,
            _utils.sanitized_type(type_hints.get(name, Any)),
            sanitized_default_value(value),
        )
        for name, value in kwargs_for_target.items()
    }

    if populate_full_signature is True:
        # Populate dataclass fields based on the target's signature.
        #
        # A user-specified parameter value (via `kwargs_for_target`) takes precedent over
        # the default-value from the signature

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
            if n + 1 <= len(pos_args):
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
            elif param.name in fields_set_by_bases:
                # don't populate a parameter that can be derived from a base
                continue
            else:
                # any parameter whose default value is None is automatically
                # annotated with `Optional[...]`. This improves flexibility with
                # Hydra's type-validation
                param_field = (
                    param.name,
                    _utils.sanitized_type(
                        type_hints.get(param.name, Any),
                        wrap_optional=param.default is None,
                    ),
                )

                if param.default is inspect.Parameter.empty:
                    if not hydra_partial:
                        # No default value specified in signature or by the user.
                        # We don't include these fields if the user specified a partial build
                        # because we assume that they want to fill these in by using partial
                        base_fields.append(param_field)
                else:
                    param_field += (sanitized_default_value(param.default),)
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

    if dataclass_name is None:
        if hydra_partial is False:
            dataclass_name = f"Builds_{target.__name__}"
        else:
            dataclass_name = f"PartialBuilds_{target.__name__}"

    out = make_dataclass(
        dataclass_name, fields=base_fields, bases=builds_bases, frozen=frozen
    )

    if hydra_partial is False and hasattr(out, _PARTIAL_TARGET_FIELD_NAME):
        # `out._partial_target_` has been inherited; this will lead to an error when
        # hydra-instantiation occurs, since it will be passed to target.
        # There is not an easy way to delete this, since it comes from a parent class
        raise TypeError(
            _utils.building_error_prefix(target)
            + "`builds(..., hydra_partial=False, builds_bases=(...))` does not "
            "permit `builds_bases` where a partial target has been specified."
        )

    out.__doc__ = (
        f"A structured config designed to {'partially ' if hydra_partial else ''}initialize/call "
        f"`{_utils.get_obj_path(target)}` upon instantiation by hydra."
    )
    if hasattr(target, "__doc__"):
        target_doc = target.__doc__
        if target_doc:
            out.__doc__ += (
                f"\n\nThe docstring for {_utils.safe_name(target)} :\n\n" + target_doc
            )
    return out
