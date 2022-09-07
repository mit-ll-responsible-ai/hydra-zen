# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import Counter
from dataclasses import (  # use this for runtime checks
    MISSING,
    Field as _Field,
    InitVar,
    dataclass,
    make_dataclass,
)
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from typing_extensions import Literal

from hydra_zen._compatibility import PATCH_OMEGACONF_830
from hydra_zen.structured_configs import _utils
from hydra_zen.structured_configs._implementations import (
    _BUILDS_CONVERT_SETTINGS,
    sanitize_collection,
)
from hydra_zen.typing import SupportedPrimitive
from hydra_zen.typing._implementations import (
    AllConvert,
    DataClass,
    DataClass_,
    DefaultsList,
    Field,
    ZenConvert,
)

from .._compatibility import HYDRA_SUPPORTS_PARTIAL
from ._globals import (
    CONVERT_FIELD_NAME,
    DEFAULTS_LIST_FIELD_NAME,
    PARTIAL_FIELD_NAME,
    RECURSIVE_FIELD_NAME,
    ZEN_PARTIAL_FIELD_NAME,
    ZEN_TARGET_FIELD_NAME,
)
from ._implementations import _retain_type_info, builds, sanitized_field
from ._type_guards import uses_zen_processing

__all__ = ["ZenField", "make_config"]


class NOTHING:
    def __init__(self) -> None:
        raise TypeError("`NOTHING` cannot be instantiated")


@dataclass
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

    hint: type = Any
    default: Union[SupportedPrimitive, Field[Any]] = _utils.field(default=NOTHING)
    name: Union[str, Type[NOTHING]] = NOTHING
    zen_convert: InitVar[Optional[ZenConvert]] = None
    _permit_default_factory: InitVar[bool] = True

    def __post_init__(
        self, zen_convert: InitVar[Optional[ZenConvert]], _permit_default_factory: bool
    ) -> None:
        if not isinstance(self.name, str):
            if self.name is not NOTHING:
                raise TypeError(f"`ZenField.name` expects a string, got: {self.name}")
        convert_settings = _utils.merge_settings(zen_convert, _BUILDS_CONVERT_SETTINGS)
        del zen_convert

        self.hint = _utils.sanitized_type(self.hint)

        if self.default is not NOTHING:
            self.default = sanitized_field(
                self.default,
                _mutable_default_permitted=_permit_default_factory,
                convert_dataclass=convert_settings["dataclass"],
            )


def _repack_zenfield(
    value: ZenField,
    name: str,
    bases: Tuple[DataClass_, ...],
    zen_convert: ZenConvert,
):
    default = value.default

    if (
        PATCH_OMEGACONF_830
        and bases
        and not _utils.mutable_default_permitted(bases, field_name=name)
        and isinstance(default, _Field)
        and default.default_factory is not MISSING
    ):  # pragma: no cover
        return ZenField(
            hint=value.hint,
            default=default.default_factory(),
            name=value.name,
            _permit_default_factory=False,
            zen_convert=zen_convert,
        )
    return value


_MAKE_CONFIG_SETTINGS = AllConvert(dataclass=False)


def make_config(
    *fields_as_args: Union[str, ZenField],
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    hydra_defaults: Optional[DefaultsList] = None,
    config_name: str = "Config",
    frozen: bool = False,
    bases: Tuple[Type[DataClass_], ...] = (),
    zen_convert: Optional[ZenConvert] = None,
    **fields_as_kwargs: Union[SupportedPrimitive, ZenField],
) -> Type[DataClass]:
    """
    Returns a config with user-defined field names and, optionally,
    associated default values and/or type annotations.

    Unlike `builds`, `make_config` is not used to configure a particular target
    object; rather, it can be used to create more general configs [1]_.

    Parameters
    ----------
    *fields_as_args : str | ZenField
        The names of the fields to be be included in the config. Or, `ZenField`
        instances, each of which details the name and their default value and/or the
        type annotation of a given field.

    **fields_as_kwargs : SupportedPrimitive | ZenField
        Like ``fields_as_args``, but field-name/default-value pairs are
        specified as keyword arguments. `ZenField` can also be used here
        to express a field's type-annotation and/or its default value.

        Named parameters of the forms ``hydra_xx``, ``zen_xx``, and ``_zen_xx`` are reserved to ensure future-compatibility, and cannot be specified by the user.

    zen_convert : Optional[ZenConvert]
        A dictionary that modifies hydra-zen's value and type conversion behavior.
        Consists of the following optional key-value pairs (:ref:`zen-convert`):

        - `dataclass` : `bool` (default=False):
            If `True` any dataclass type/instance without a
            `_target_` field is automatically converted to a targeted config
            that will instantiate to that type/instance. Otherwise the dataclass
            type/instance will be passed through as-is.

    bases : Tuple[Type[DataClass], ...], optional (default=())
        Base classes that the resulting config class will inherit from.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how Hydra handles the non-primitive objects passed to configuration [3]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.


    hydra_defaults : None | list[str | dict[str, str | list[str] | None ]], optional (default = None)
        A list in an input config that instructs Hydra how to build the output config
        [7]_ [8]_. Each input config can have a Defaults List as a top level element. The
        Defaults List itself is not a part of output config.

    frozen : bool, optional (default=False)
        If ``True``, the resulting config class will produce 'frozen' (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance of the config will raise
        :py:class:`dataclasses.FrozenInstanceError` at runtime.

    config_name : str, optional (default="Config")
        The class name of the resulting config class.

    Returns
    -------
    Config : Type[DataClass]
        The resulting config class; a dataclass that possess the user-specified fields.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError
        The provided configured value cannot be serialized by Hydra, nor does hydra-zen
        provide specialized support for it. See :ref:`valid-types` for more details.

    Notes
    -----
    The resulting "config" is a dataclass-object [4]_ with Hydra-specific attributes
    attached to it, along with the attributes specified via ``fields_as_args`` and
    ``fields_as_kwargs``.

    Any field specified without a type-annotation is automatically annotated with
    :py:class:`typing.Any`. Hydra only supports a narrow subset of types [5]_;
    `make_config` will automatically 'broaden' any user-specified annotations so that
    they are compatible with Hydra.

    `make_config` will automatically manipulate certain types of default values to
    ensure that they can be utilized in the resulting config and by Hydra:

    - Mutable default values will automatically be packaged in a default factory function [6]_
    - A default value that is a class-object or function-object will automatically be wrapped by `just`, to ensure that the resulting config is serializable by Hydra.

    For finer-grain control over how type annotations and default values are managed,
    consider using :func:`dataclasses.make_dataclass`.

    For details of the annotation `SupportedPrimitive`, see :ref:`valid-types`.

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.
    just : Create a config that "just" returns a class-object or function, without instantiating/calling it.

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [4] https://docs.python.org/3/library/dataclasses.html
    .. [5] https://hydra.cc/docs/tutorials/structured_config/intro/#structured-configs-supports
    .. [6] https://docs.python.org/3/library/dataclasses.html#default-factory-functions
    .. [7] https://hydra.cc/docs/tutorials/structured_config/defaults/
    .. [8] https://hydra.cc/docs/advanced/defaults_list/

    Examples
    --------
    >>> from hydra_zen import make_config, to_yaml
    >>> def pp(x):
    ...     return print(to_yaml(x))  # pretty-print config as yaml

    **Basic Usage**

    Let's create a bare-bones config with two fields, named 'a' and 'b'.

    >>> Conf1 = make_config("a", "b")  # sig: `Conf(a: Any, b: Any)`
    >>> pp(Conf1)
    a: ???
    b: ???

    Now we'll configure these fields with particular values:

    >>> conf1 = Conf1(1, "hi")
    >>> pp(conf1)
    a: 1
    b: hi
    >>> conf1.a
    1
    >>> conf1.b
    'hi'

    We can also specify fields via keyword args; this is especially convenient
    for providing associated default values.

    >>> Conf2 = make_config("unit", data=[-10, -20])
    >>> pp(Conf2)
    unit: ???
    data:
    - -10
    - -20

    Configurations can be nested

    >>> Conf3 = make_config(c1=Conf1(a=1, b=2), c2=Conf2)
    >>> pp(Conf3)
    c1:
      a: 1
      b: 2
    c2:
      unit: ???
      data:
      - -10
      - -20
    >>> Conf3.c1.a
    1

    Configurations can be composed via inheritance

    >>> ConfInherit = make_config(c=2, bases=(Conf2, Conf1))
    >>> pp(ConfInherit)
    a: ???
    b: ???
    unit: ???
    data:
    - -10
    - -20
    c: 2

    >>> issubclass(ConfInherit, Conf1) and issubclass(ConfInherit, Conf2)
    True

    **Support for Additional Types**

    Types like :py:class:`complex` and :py:class:`pathlib.Path` are automatically
    supported by hydra-zen.

    >>> ConfWithComplex = make_config(a=1+2j)
    >>> pp(ConfWithComplex)
    a:
      real: 1.0
      imag: 2.0
      _target_: builtins.complex

    See :ref:`additional-types` for a complete list of supported types.

    **Using ZenField to Provide Type Information**

    The `ZenField` class can be used to include a type-annotation in association
    with a field.

    >>> from hydra_zen import ZenField as zf
    >>> ProfileConf = make_config(username=zf(str), age=zf(int))
    >>> # signature: ProfileConf(username: str, age: int)

    Providing type annotations is optional, but doing so enables Hydra to perform
    checks at runtime to ensure that a configured value matches its associated
    type [4]_.

    >>> pp(ProfileConf(username="piro", age=False))  # age should be an integer
    <ValidationError: Value 'False' could not be converted to Integer>

    These default values can be provided alongside type annotations

    >>> C = make_config(age=zf(int, 0))  # signature: C(age: int = 0)

    `ZenField` can also be used to specify ``fields_as_args``; here, field names
    must be specified as well.

    >>> C2 = make_config(zf(name="username", hint=str), age=zf(int, 0))
    >>> # signature: C2(username: str, age: int = 0)

    See :ref:`data-val` for more general data validation capabilities via hydra-zen.
    """
    convert_settings = _utils.merge_settings(zen_convert, _MAKE_CONFIG_SETTINGS)
    convert_settings = cast(ZenConvert, convert_settings)
    del zen_convert

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
        all_names = [f.name if isinstance(f, ZenField) else f for f in fields_as_args]
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

        if not isinstance(_defaults, ZenField):
            hydra_defaults = _defaults  # type: ignore

    # validate hydra-args via `builds`
    # also check for use of reserved names
    builds(
        dict,
        hydra_convert=hydra_convert,
        hydra_recursive=hydra_recursive,
        hydra_defaults=hydra_defaults,
        **{k: None for k in fields_as_kwargs},
    )

    normalized_fields: Dict[str, ZenField] = {}

    for _field in fields_as_args:
        if isinstance(_field, str):
            normalized_fields[_field] = ZenField(
                name=_field, hint=Any, zen_convert=convert_settings
            )
        else:
            assert isinstance(_field.name, str)
            normalized_fields[_field.name] = _repack_zenfield(
                _field, _field.name, bases, convert_settings
            )

    for name, value in fields_as_kwargs.items():
        if not isinstance(value, ZenField):
            default_factory_permitted = (
                not bases or _utils.mutable_default_permitted(bases, field_name=name)
                if PATCH_OMEGACONF_830
                else True
            )
            normalized_fields[name] = ZenField(
                name=name,
                default=value,
                _permit_default_factory=default_factory_permitted,
                zen_convert=convert_settings,
            )
        else:
            normalized_fields[name] = _repack_zenfield(
                value, name=name, bases=bases, zen_convert=convert_settings
            )

    # fields without defaults must come first
    config_fields: List[Union[Tuple[str, type], Tuple[str, type, Any]]] = [
        (str(f.name), f.hint)
        for f in normalized_fields.values()
        if f.default is NOTHING
    ]

    config_fields.extend(
        [
            (
                str(f.name),
                (
                    # f.default: Field
                    # f.default.default: Any
                    f.hint
                    if _retain_type_info(
                        type_=f.hint,
                        value=f.default.default,  # type: ignore
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
            (CONVERT_FIELD_NAME, str, _utils.field(default=hydra_convert, init=False))
        )

    if hydra_defaults is not None:
        hydra_defaults = sanitize_collection(hydra_defaults, convert_dataclass=False)
        config_fields.append(
            (
                DEFAULTS_LIST_FIELD_NAME,
                List[Any],
                _utils.field(default_factory=lambda: list(hydra_defaults), init=False),
            )
        )

    out = make_dataclass(
        cls_name=config_name, fields=config_fields, frozen=frozen, bases=bases
    )
    if hasattr(out, ZEN_TARGET_FIELD_NAME) and not uses_zen_processing(out):
        raise ValueError(
            f"{out.__name__} inherits from base classes that overwrite some fields "
            f"associated with zen-processing features. As a result, this config will "
            f"not instantiate correctly."
        )
    if (
        HYDRA_SUPPORTS_PARTIAL
        and getattr(out, PARTIAL_FIELD_NAME, False)
        and uses_zen_processing(out)
    ):
        raise ValueError(
            f"{out.__name__} specifies both `{PARTIAL_FIELD_NAME}=True` and `"
            f"{ZEN_PARTIAL_FIELD_NAME}=True`. This config will not instantiate "
            f"correctly. This is typically caused by inheriting from multiple, "
            f"conflicting configs."
        )

    return cast(Type[DataClass], out)
