# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import Optional, Union

from typing_extensions import Literal

from hydra_zen.typing import DataclassOptions, SupportedPrimitive
from hydra_zen.typing._implementations import (
    AllConvert,
    DataClass,
    DataClass_,
    DefaultsList,
    ZenConvert,
)

from ._implementations import DefaultBuilds, ZenField

__all__ = ["ZenField", "make_config"]


class NOTHING:
    def __init__(self) -> None:
        raise TypeError("`NOTHING` cannot be instantiated")


_MAKE_CONFIG_SETTINGS = AllConvert(dataclass=False, flat_target=False)


def make_config(
    *fields_as_args: Union[str, ZenField],
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    hydra_defaults: Optional[DefaultsList] = None,
    zen_dataclass: Optional[DataclassOptions] = None,
    bases: tuple[type[DataClass_], ...] = (),
    zen_convert: Optional[ZenConvert] = None,
    **fields_as_kwargs: Union[SupportedPrimitive, ZenField],
) -> type[DataClass]:
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

    hydra_convert : Optional[Literal["none", "partial", "all", "object"]], optional (default="none")
        Determines how Hydra handles the non-primitive objects passed to configuration [3]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without a trace of OmegaConf containers
        - ``"object"``: Passed objects are converted to dict and list. Structured Configs are converted to instances of the backing dataclass / attr class.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    hydra_defaults : None | list[str | dict[str, str | list[str] | None ]], optional (default = None)
        A list in an input config that instructs Hydra how to build the output config
        [7]_ [8]_. Each input config can have a Defaults List as a top level element. The
        Defaults List itself is not a part of output config.

    zen_dataclass : Optional[DataclassOptions]
        A dictionary can specify any option that is supported by
        :py:func:`dataclasses.make_dataclass` other than `fields`.

        Additionally, a ``'module': <str>`` entry can be specified to enable pickle
        compatibility. See `hydra_zen.typing.DataclassOptions` for details.

    frozen : bool, optional (default=False)
        .. deprecated:: 0.9.0
            `frozen` will be removed in hydra-zen 0.10.0. It is replaced by
            ``zen_dataclass={'frozen': <bool>}``.

        If ``True``, the resulting config class will produce 'frozen' (i.e. immutable)
        instances. I.e. setting/deleting an attribute of an instance of the config will
        raise :py:class:`dataclasses.FrozenInstanceError` at runtime.

    config_name : str, optional (default="Config")
        .. deprecated:: 0.9.0
            `config_name` will be removed in hydra-zen 0.10.0. It is replaced by
            ``zen_dataclass={'cls_name': <str>}``.

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
    ``fields_as_kwargs``. **Unlike std-lib dataclasses, the default value for
    unsafe_hash is True.**

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
    >>> Conf3().c1.a
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

    >>> issubclass(ConfInherit, Conf1) and issubclass(ConfInherit, Conf2)  # type: ignore
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
    _locals = locals().copy()
    fields_as_args = _locals.pop("fields_as_args")
    fields_as_kwargs = _locals.pop("fields_as_kwargs")
    return DefaultBuilds.make_config(*fields_as_args, **_locals, **fields_as_kwargs)  # type: ignore
