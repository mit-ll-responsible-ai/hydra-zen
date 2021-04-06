import inspect
from collections import defaultdict
from dataclasses import Field, dataclass, field, fields, is_dataclass, make_dataclass
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    get_type_hints,
    overload,
)

from typing_extensions import Final, Literal

from hydra_utils.funcs import identity, partial
from hydra_utils.structured_configs import _utils
from hydra_utils.typing import Builds, Importable, Instantiable, Just, PartialBuilds

__all__ = ["builds", "just", "hydrated_dataclass", "mutable_value"]

_TARGET_FIELD_NAME: Final[str] = "_target_"
_PARTIAL_TARGET_FIELD_NAME: Final[str] = "_partial_target_"


def _check_importable_path(obj: Any):
    """
    Raises if `obj` is defined in unreachable namespace

    Parameters
    ----------
    obj : Any

    Raises
    ------
    ModuleNotFoundError

    Examples
    --------
    >>> class C:
    ...     def f(self): pass
    >>> _check_importable_path(C)  # OK
    >>> _check_importable_path(C.f)  # raises
    """
    path = _utils.get_obj_path(obj)
    # catches "<locals>" and "<unknown>"
    if "<" in path:
        name = _utils.safe_name(obj)
        raise ModuleNotFoundError(
            f"{name} is not importable from path: {_utils.get_obj_path(obj)}"
        )


def mutable_value(x: Any) -> Field:
    """Used to set a mutable object as a default value for a field
    in a dataclass.

    This is an alias for ``field(default_factory=lambda: x)``

    Examples
    --------
    >>> from hydra_utils import mutable_value
    >>> from dataclasses import dataclass

    See https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    >>> @dataclass
    ... class HasMutableDefault
    ...     a_list: list  = [1, 2, 3]  # error: mutable default

    Using `mutable_value` to specify the default list:

    >>> @dataclass
    ... class HasMutableDefault
    ...     a_list: list  = mutable_value([1, 2, 3])  # ok"""
    return field(default_factory=lambda: x)


Field_Entry = Tuple[str, type, Field]


class hydrated_dataclass:
    """A decorator that uses `hydra_utils.builds` to create a dataclass with the appropriate
    hydra-specific fields for specifying a structured config.

    Examples
    --------
    A simple usage of `hydrated_dataclass`. Here, we specify a structured config

    >>> from hydra_utils import hydrated_dataclass, instantiate
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

    def __init__(
        self,
        target: Importable,
        *,
        populate_full_signature: bool = False,
        hydra_partial: bool = False,
        hydra_recursive: bool = True,
        hydra_convert: Literal["none", "partial", "all"] = "none",
    ):
        """
        Parameters
        ----------
        target : Union[Instantiable, Callable]
            The object to be instantiated/called.

        populate_full_signature : bool, optional (default=False)
            If True, then the resulting dataclass's __init__ signature and fields
            will be populated according to the signature of ``target``.

            Values specified in **kwargs_for_target take precedent over the corresponding
            default values from the signature.

        hydra_partial : bool, optional (default=False)
            If True, then hydra-instantiation produces `functools.partial(target, **kwargs)`

        hydra_recursive : bool, optional (default=True)
            If True, then upon hydra will recursively instantiate all other
            hydra-config objects nested within this dataclass [2]_.

        hydra_convert: Literal["none", "partial", "all"], optional (default="none")
            Determines how hydra handles the non-primitive objects passed to `target` [3]_.

               none - Passed objects are DictConfig and ListConfig, default
            partial - Passed objects are converted to dict and list, with
                      the exception of Structured Configs (and their fields).
                all - Passed objects are dicts, lists and primitives without
                      a trace of OmegaConf containers
        """
        self._target = target
        self._populate_full_signature = populate_full_signature
        self._hydra_recursive = hydra_recursive
        self._hydra_convert = hydra_convert
        self._hydra_partial = hydra_partial

    def __call__(self, decorated_obj: type) -> Builds[Importable]:
        if not isinstance(decorated_obj, type):
            raise NotImplementedError(
                "Class instances are not supported by `hydrated_dataclass` (yet)."
            )

        # TODO: We should mutate `decorated_obj` directly like @dataclass does.
        #       Presently, we create an intermediate dataclass that we inherit
        #       from, which gets the job done for the most part but there are
        #       practical differences. E.g. you cannot delete an attribute that
        #       was declared in the definition of `decorated_obj`.
        decorated_obj = dataclass(decorated_obj)

        return builds(
            self._target,
            populate_full_signature=self._populate_full_signature,
            hydra_recursive=self._hydra_recursive,
            hydra_convert=self._hydra_convert,
            hydra_partial=self._hydra_partial,
            builds_bases=(decorated_obj,),
            dataclass_name=decorated_obj.__name__,
        )


def just(obj: Importable) -> Just[Importable]:
    """Produces a structured config that, when instantiated by hydra, 'just'
    returns ``obj``.

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
    Demonstrating how to use ``just`` to specify a particular torch-optimizer
    in your structured config:

    >>> from torch.optim import Adam
    >>> from hydra_utils import just
    >>> @dataclass
    ... class ModuleConfig:
    ...     optimizer: Any = just(Adam)

    Demonstrating the simple behavior of ``just`` in the context of leveraging ``hydra``.

    >>> from hydra_utils import just, instantiate
    >>> just_str_conf = just(str)
    >>> str is instantiate(just_str_conf)  # "just" returns the object `str`
    True
    >>> just_str_conf._target_
    'hydra_utils.funcs.identity'
    >>> just_str_conf.obj
    '${get_obj:builtins.str}'
    """
    try:
        obj_path = _utils.get_obj_path(obj)
    except AttributeError:
        raise AttributeError(
            f"`just({obj})`: `obj` is not importable; it is missing the attributes `__module__` and/or `__qualname__`"
        )

    _check_importable_path(obj)

    out_class = make_dataclass(
        ("Just_" + _utils.safe_name(obj)),
        [
            (
                _TARGET_FIELD_NAME,
                str,
                field(default=_utils.get_obj_path(identity), init=False),
            ),
            (
                "obj",
                Any,
                field(
                    default=_utils.interpolated("hydra_utils_get_obj", obj_path),
                    init=False,
                ),
            ),
        ],
    )
    out_class.__doc__ = (
        f"A structured config designed to return {obj} when it is instantiated by hydra"
    )

    return out_class


# overloads when `hydra_partial=False`
@overload
def builds(
    target: Importable,
    *,
    populate_full_signature: bool = False,
    hydra_partial: Literal[False] = False,
    hydra_recursive: bool = True,
    hydra_convert: Literal["none", "partial", "all"] = "none",
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    **kwargs_for_target,
) -> Builds[Importable]:  # pragma: no cover
    ...


# overloads when `hydra_partial=True`
@overload
def builds(
    target: Importable,
    *,
    populate_full_signature: bool = False,
    hydra_partial: Literal[True],
    hydra_recursive: bool = True,
    hydra_convert: Literal["none", "partial", "all"] = "none",
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    **kwargs_for_target,
) -> PartialBuilds[Importable]:  # pragma: no cover
    ...


# overloads when `hydra_partial: bool`
@overload
def builds(
    target: Importable,
    *,
    populate_full_signature: bool = False,
    hydra_partial: bool,
    hydra_recursive: bool = True,
    hydra_convert: Literal["none", "partial", "all"] = "none",
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    **kwargs_for_target,
) -> Union[Builds[Importable], PartialBuilds[Importable]]:  # pragma: no cover
    ...


def builds(
    target: Importable,
    *,
    populate_full_signature: bool = False,
    hydra_partial: bool = False,
    hydra_recursive: bool = True,
    hydra_convert: Literal["none", "partial", "all"] = "none",
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    **kwargs_for_target,
) -> Union[Builds[Importable], PartialBuilds[Importable]]:
    """Produces a structured config (i.e. a dataclass) [1]_ that, when instantiated by hydra,
    initializes/calls ``target`` with the provided keyword arguments.

    The returned object is an un-instantiated dataclass.

    ``builds`` provides a simple and functional way to dynamically create rich structured
    configurations.

    Parameters
    ----------
    target : Union[Instantiable, Callable]
        The object to be instantiated/called

    **kwargs_for_target : Any
        The keyword arguments passed to `target(...)`.

        The arguments specified here solely determine the fields and init-parameters of the
        resulting dataclass, unless ``populate_full_signature=True`` is specified (see below).

    populate_full_signature : bool, optional (default=False)
        If True, then the resulting dataclass's __init__ signature and fields
        will be populated according to the signature of ``target``.

        Values specified in **kwargs_for_target take precedent over the corresponding
        default values from the signature.

    hydra_partial : bool, optional (default=False)
        If True, then hydra-instantiation produces `functools.partial(target, **kwargs)`

    hydra_recursive : bool, optional (default=True)
        If True, then upon hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

    hydra_convert: Literal["none", "partial", "all"], optional (default="none")
        Determines how hydra handles the non-primitive objects passed to `target` [3]_.

           none - Passed objects are DictConfig and ListConfig, default
        partial - Passed objects are converted to dict and list, with
                  the exception of Structured Configs (and their fields).
            all - Passed objects are dicts, lists and primitives without
                  a trace of OmegaConf containers

    builds_bases : Tuple[DataClass, ...]

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
        are not compatible with the signature of ``target``.

    Notes
    -----
    ``builds`` provides runtime validation of user-specified named arguments against
    the target's signature. This helps to ensure that typos in field names fail
    early and explicitly.

    Mutable values are automatically specified using ``field(default_factory=lambda: <value>)`` [4]_.

    Type annotations are inferred from the target's signature and are only retained if they are compatible
    with hydra's limited set of supported annotations; otherwise `Any` is specified.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [4] https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    Examples
    --------
    **Basic Usage**

    >>> from hydra_utils import builds, instantiate
    >>> builds(dict, a=1, b='x')  # makes a dataclass that will "build" a dictionary with the specified fields
    types.Builds_dict
    >>> instantiate(builds(dict, a=1, b='x'))  # using hydra to build the dictionary
    {'a': 1, 'b': 'x'}

    Using ``builds`` with partial instantiation

    >>> def a_two_tuple(x: int, y: float): return x, y
    >>> PartialBuildsFunc = builds(a_two_tuple, x=1, hydra_partial=True)  # specifies only `x`
    >>> partial_func = instantiate(PartialBuildsFunc)
    >>> partial_func
    functools.partial(<function a_two_tuple at 0x00000220A7820EE0>, x=1)
    >>> partial_func(y=22)
    (1, 22)

    Using ``builds`` makes it easy to compose configurations:

    >>> import numpy as np
    >>> LoadsData = builds(np.array, object=[1., 2.])
    >>> BuildsDict = builds(dict, data=LoadsData, units="meters")
    >>> instantiate(BuildsDict)
    {'data': array([1., 2.]), 'units': 'meters'}


    **Understanding What builds is Doing**

    It is important to gain some insight into the dataclass that ``builds`` creates.
    Let's dig a bit deeper into the first example.

    >>> from dataclasses import is_dataclass
    >>> import inspect
    >>> DictConf = builds(dict, a=1, b='x')  # creates a (uninstantiated) dataclass
    >>> DictConf
    types.Builds_dict
    >>> is_dataclass(DictConf)
    True
    >>> inspect.signature(DictConf)  # the dataclass' signature reflects the arguments passed to ``builds``
    <Signature (a: Any = 1, b: Any = 'x') -> None>
    >>> DictConf.a  # class attribute of `Builds_dict`
    1
    >>> DictConf.b  # class attribute of `Builds_dict`
    "x"

    We can create an instance ``DictConf`` dataclass to overrides its default values

    >>> DictConf(a=-10)  # creates an instance of `DictConf`
    Builds_dict(_target_='builtins.dict', _recursive_=True, _convert_='none', a=-10, b='x')
    >>> instantiate(DictConf(a=-10))
    {'a': -10, 'b': 'x'}

    What is going on under the hood is that ``builds`` is defining a dataclass that is compatible with
    hydra's mechanism for instantiating/calling objects; the dataclass can be serialized to, and recreated from, a yaml:

    >>> from omegaconf import OmegaConf
    >>> print(OmegaConf.to_yaml(DictConf))
    _target_: builtins.dict
    _recursive_: true
    _convert_: none
    a: 1
    b: x

    ``_target_``, ``_recursive_``, and ``_convert_`` are hydra-specific fields that are automatically created by
    ``builds``. These are be controlled via ``builds(<target>, hydra_convert=<>, hydra_recursive=<>)``.

    **Additional Features and Functionality**

    Auto-populating the fields of the dataclass using the target's signature:

    >>> def a_function(a: int, b: str, c: float=-10.): return (a, b, c)
    >>> inspect.signature(builds(a_function)) # no arguments are specified -> signature is empty by default
    <Signature () -> None>
    >>> builds_a_function = builds(a_function, populate_full_signature=True)  # auto-populate signature
    >>> inspect.signature(builds_a_function)  # hydra-compatible annotations and default values are preserved
    <Signature (a: int, b: str, c: float = -10.0) -> None>
    >>> instantiate(builds_a_function(a=1, b="hi"))
    (1, 'hi', -10.0)

    You can auto-populate your structured config

    ``builds`` will raise if you specify an argument that is incompatible with the target. This means
    that you will catch mistakes before you try to instantiate your configurations

    >>> builds(a_function, z=10)
    TypeError: Building: a_function ..
    The following unexpected keyword argument(s) was specified for __main__.a_function via `builds`: z

    **Some Examples of Using builds for Configuring ML Workflows**

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear

    Demonstrating how to use ``builds`` to configure a torch-optimizer with a
    specified learning rate in your structured config:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class ModuleConfig:
    ...     # hydra-instantiation produces a config where `config.optimizer`
    ...     # is: `functools.partial(Adam, lr=1e-5)`
    ...     optimizer: Any = builds(Adam, lr=1e-5, hydra_partial=True)
    ...     model: Any = builds(Linear, in_features=10, out_features=2)

    Note that omegaconf/hydra-style interpolation works too:

    >>> @dataclass
    ... class ModuleConfig:
    ...     # hydra-instantiation produces a config where `config.optimizer`
    ...     # is: `functools.partial(Adam, lr=10.2)`
    ...     learning_rate : float = 10.2
    ...     in_features : int = 10
    ...     out_features : int = 2
    ...     optimizer: Any = builds(Adam, lr="${learning_rate}", hydra_partial=True)
    ...     model: Any = builds(Linear, in_features="${in_features}", out_features="${out_features}")

    >>> instantiate(ModuleConfig)
    {'learning_rate': 10.2,
     'in_features': 10,
     'out_features': 2,
     'optimizer': functools.partial(<class 'torch.optim.adam.Adam'>, lr=10.2),
     'model': Linear(in_features=10, out_features=2, bias=True)}
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

    if not isinstance(hydra_recursive, bool):
        raise TypeError(
            f"`hydra_recursive` must be a boolean type, got {hydra_recursive}"
        )

    if not isinstance(hydra_partial, bool):
        raise TypeError(f"`hydra_partial` must be a boolean type, got: {hydra_partial}")

    if not isinstance(hydra_convert, str):
        raise TypeError(
            f"`hydra_convert` must be 'none', 'partial', or 'all', got: {hydra_convert}"
        )

    hydra_convert = hydra_convert.lower()  # normalize casing

    if hydra_convert not in {"none", "partial", "all"}:
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

    _check_importable_path(target)

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

    # `base_fields` stores the list of fields that will be present in our dataclass
    #
    # Note: _args_ should be added here once hydra supports it in structured configs
    # TODO: should we always write these? Or is it more legible to only write them
    #       if they were explicitly specified by the user?
    #       - Presently we always need to write these, otherwise inheritance
    #         becomes an issue (as it is with _partial_target_
    base_fields: List[Union[Tuple[str, type], Field_Entry]] = target_field + [
        ("_recursive_", bool, field(default=hydra_recursive, init=False)),
        ("_convert_", str, field(default=hydra_convert, init=False)),
    ]

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
        type_hints = get_type_hints(target)
    except TypeError:
        # Covers case for ufuncs, which do not have inspectable type hints
        type_hints = defaultdict(lambda: Any)

    # True if **kwargs is in the signature of `target`
    sig_has_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature_params.values()
    )

    # these are the names of the only parameters in the signature of `target` that can
    # be referenced by name
    nameable_params_in_sig = set(
        p.name
        for p in signature_params.values()
        if p.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    )

    fields_set_by_bases: Set[str] = {
        _field.name
        for _base in builds_bases
        for _field in fields(_base)
        if not _field.name.startswith("_")
    }

    # Validate that user-specified arguments correspond to parameters in target's
    # signature that can be specified by name
    if target_has_valid_signature and not sig_has_var_kwargs:
        if not set(kwargs_for_target) <= nameable_params_in_sig:
            _unexpected = set(kwargs_for_target) - nameable_params_in_sig
            raise TypeError(
                _utils.building_error_prefix(target)
                + f"The following unexpected keyword argument(s) was specified for {_utils.get_obj_path(target)} via `builds`: "
                f"{', '.join(_unexpected)}"
            )
        if not fields_set_by_bases <= nameable_params_in_sig:
            _unexpected = fields_set_by_bases - nameable_params_in_sig
            raise TypeError(
                _utils.building_error_prefix(target)
                + f"The following unexpected keyword argument(s) for {_utils.get_obj_path(target)} "
                f"was specified via inheritance from a base class: "
                f"{', '.join(_unexpected)}"
            )

    # Create valid dataclass fields from the user-specified values
    #
    # user_specified_params: arg-name -> (arg-name, arg-type, field-w-value)
    #  - arg-type: taken from the parameter's annotation in the target's signature
    #    and is resolved to one of the type-annotations supported by hydra if possible,
    #    otherwise, is Any
    #  - arg-value: mutable values are automatically specified using default-factory
    user_specified_params: Dict[str, Tuple[str, type, Field]] = {
        name: (
            name,
            _utils.sanitized_type(type_hints.get(name, Any)),
            (
                field(default=value)
                if not isinstance(value, _utils.KNOWN_MUTABLE_TYPES)
                else mutable_value(value)
            ),
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
        for param in signature_params.values():
            if param.name not in nameable_params_in_sig:
                # parameter cannot be specified by name
                continue

            if param.name in user_specified_params:
                # user-specified parameter is preferred
                _fields_with_default_values.append(user_specified_params[param.name])
                _seen.add(param.name)
            elif param.name in fields_set_by_bases:
                # don't populate a parameter that can be derived from a base
                continue
            else:
                param_field = (
                    param.name,
                    _utils.sanitized_type(type_hints.get(param.name, Any)),
                )

                if param.default is inspect.Parameter.empty:
                    # no default-value specified in signature
                    base_fields.append(param_field)
                else:
                    if isinstance(param.default, _utils.KNOWN_MUTABLE_TYPES):
                        value = mutable_value(param.default)
                    else:
                        value = field(default=param.default)
                    param_field += (value,)
                    _fields_with_default_values.append(param_field)

        base_fields.extend(_fields_with_default_values)

        if sig_has_var_kwargs:
            # if the signature has **kwargs, then we need to add any user-specified
            # parameters that have not already been added
            base_fields.extend(
                entry
                for name, entry in user_specified_params.items()
                if name not in _seen
            )
    else:
        base_fields.extend(user_specified_params.values())

    if dataclass_name is None:
        if hydra_partial is False:
            dataclass_name = f"Builds_{target.__name__}"
        else:
            dataclass_name = f"PartialBuilds_{target.__name__}"

    out = make_dataclass(dataclass_name, fields=base_fields, bases=builds_bases)

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
