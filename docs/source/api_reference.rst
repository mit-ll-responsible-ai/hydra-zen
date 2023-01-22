.. meta::
   :description: Reference documentation for hydra-zen.

.. _hydra-zen-reference:

#########
Reference
#########

Encyclopedia Hydrazennica.

All reference documentation includes detailed Examples sections. Please scroll to the 
bottom of any given reference page to see the examples.

**************************************
Creating and Launching Jobs with Hydra
**************************************

hydra-zen provides users the ability to launch a Hydra job via a 
Python function instead of from a commandline interface.

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   launch
   zen
   wrapper.Zen

*********************************
Creating and Working with Configs
*********************************

hydra-zen provides us with some simple but powerful tools for creating and working with 
configs. Among these, the most essential functions for creating configs are 
:func:`~hydra_zen.make_config` and :func:`~hydra_zen.builds`. Then, 
:func:`~hydra_zen.instantiate` can be used to resolve these configs so that they return 
the data and class-instances that we need for our application.

.. _create-config:

Creating Configs
****************
.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   make_config
   builds
   just
   hydrated_dataclass


Storing Configs
***************

.. toctree::

   store

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   ZenStore
   wrapper.default_to_config
   


Instantiating and Resolving Configs
***********************************

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   instantiate
   get_target


Utilities
*********

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/


   make_custom_builds_fn
   is_partial_builds
   uses_zen_processing
   ZenField


Working with YAMLs
******************
Hydra serializes all configs to a YAML-format text when launching a job.
The following utilities can be used to work with YAML-serialized configs.

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/


   to_yaml
   save_as_yaml
   load_from_yaml


hydra_zen.typing
****************
.. currentmodule:: hydra_zen.typing

.. autosummary::
   :toctree: generated/


   DataclassOptions
   ZenConvert


.. _valid-types:

**********************************************************
Configuration-Value Types Supported by Hydra and hydra-zen
**********************************************************

The types of values that can be specified in configs are limited by their ability to be 
serialized to a YAML format. hydra-zen provides automatic support for an additional set 
of common types via its :ref:`config-creation functions <create-config>`.

Types Supported Natively by Hydra
*********************************

Values of the following types can be specified directly in configs:

- ``NoneType``
- :py:class:`bool`
- :py:class:`int`
- :py:class:`float`
- :py:class:`str`
- :py:class:`list`
- :py:class:`dict`
- :py:class:`enum.Enum`
- :py:class:`bytes`  (*support added in OmegaConf 2.2.0*)
- :py:class:`pathlib.Path`  (*support added in OmegaConf 2.2.1*)

.. _additional-types:

Auto-Config Support for Additional Types via hydra-zen
******************************************************

Values of additional types can be specified directly via hydra-zen's 
:ref:`config-creation functions <create-config>` (and other utility functions like :func:`~hydra_zen.to_yaml`) and those functions will automatically 
create dataclass instances to represent those values in a way that is compatible 
with Hydra. For example, a :py:class:`complex` value can be passed directly to 
:func:`~hydra_zen.make_config`, and hydra-zen will generate a dataclass instance that 
represents that complex value in a Hydra-compatible way.

.. code-block:: pycon
   :caption: Demonstrating auto-config support for `complex`, `functools.partial` and `dataclasses.dataclass`

   >>> from hydra_zen import builds, just, make_config, to_yaml, instantiate
   >>> def print_yaml(x): print(to_yaml(x))

   >>> Conf = make_config(value=2.0 + 3.0j)
   >>> print_yaml(Conf)
   value:
     real: 2.0
     imag: 3.0
     _target_: builtins.complex

.. code-block:: pycon

   >>> from functools import partial

   >>> Conf2 = builds(dict, x=partial(int, 3))
   >>> print_yaml(Conf2)
   _target_: builtins.dict
   x:
     _target_: builtins.int
     _partial_: true
     _args_:
     - 3
   >>> instantiate(Conf2) 
   {'x': functools.partial(<class 'int'>, 3)}


.. code-block:: pycon

   >>> from typing import Callable, Sequence
   >>> from dataclasses import dataclass
   >>> 
   >>> @dataclass
   ... class Bar:
   ...    reduce_fn: Callable[[Sequence[float]], float] = sum
   >>>
   >>> just_bar = just(Bar())
   
   >>> print_yaml(just_bar)
   _target_: __main__.Bar
   reduce_fn:
     _target_: hydra_zen.funcs.get_obj
     path: builtins.sum
   
   >>> instantiate(just_bar)
   Bar(reduce_fn=<built-in function sum>)


Auto-config behaviors can be configured via the :ref:`zen-convert API <zen-convert>`.

hydra-zen provides specialized auto-config support for values of the following types:

- :py:class:`bytes` (*support provided for OmegaConf < 2.2.0*)
- :py:class:`bytearray`
- :py:class:`complex`
- :py:class:`collections.Counter`
- :py:class:`collections.deque`
- :py:func:`functools.partial` (note: not compatible with pickling)
- :py:class:`pathlib.Path` (*support provided for OmegaConf < 2.2.1*)
- :py:class:`pathlib.PosixPath`
- :py:class:`pathlib.WindowsPath`
- :py:class:`range`
- :py:class:`set`
- :py:class:`frozenset`
- :py:func:`dataclasses.dataclass` (note: not compatible with pickling)

hydra-zen also provides auto-config support for some third-pary libraries:

- `pydantic.dataclasses.dataclass`
- `pydantic.Field`
- `pydantic.Field`
- `torch.optim.optimizer.required` (i.e. the default parameter for `lr` in `Optimizer`)


*********************
Third-Party Utilities
*********************

.. _data-val:

Runtime Data Validation
***********************

Although Hydra provides some runtime type-checking functionality, it only supports :ref:`a limited set of types and annotations <type-support>`. hydra-zen offers support for more robust runtime type-checking capabilities via various third-party libraries.

.. currentmodule:: hydra_zen.third_party

.. autosummary::
   :toctree: generated/

   beartype.validates_with_beartype
   pydantic.validates_with_pydantic
