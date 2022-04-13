.. meta::
   :description: Reference documentation for hydra-zen.

.. _hydra-zen-reference:

#########
Reference
#########

Encyclopedia Hydrazennica.

All reference documentation includes detailed Examples sections. Please scroll to the 
bottom of any given reference page to see the examples.

*************************
Launching Jobs with Hydra
*************************

hydra-zen provides users the ability to launch a Hydra job via a 
Python function instead of from a commandline interface.

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   launch


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


.. _additional-types:

Additional Types, Supported via hydra-zen
*****************************************

Values of additional types can be specified directly via hydra-zen's 
:ref:`config-creation functions <create-config>` (and other utility functions like ``to_yaml``), and hydra-zen will automatically 
create targeted configs to represent those values in a way that is compatible 
with Hydra. For example, a :py:class:`complex` value can be specified directly via :func:`~hydra_zen.make_config`, and a targeted config will be created for that value.

.. code-block:: pycon

   >>> from hydra_zen import make_config, to_yaml
   >>> Conf = make_config(value=2.0 + 3.0j)
   >>> print(to_yaml(Conf))
   value:
     real: 2.0
     imag: 3.0
     _target_: builtins.complex
   
   >>> from functools import partial
   >>> print(to_yaml(partial(int, 3)))
   _target_: builtins.int
   _partial_: true
   _args_:
   - 3

hydra-zen provides specialized support for values of the following types:

- :py:class:`bytes`
- :py:class:`bytearray`
- :py:class:`complex`
- :py:class:`collections.Counter`
- :py:class:`collections.deque`
- :py:func:`functools.partial`  (*added in v0.5.0*)
- :py:class:`pathlib.Path`
- :py:class:`pathlib.PosixPath`
- :py:class:`pathlib.WindowsPath`
- :py:class:`range`
- :py:class:`set`
- :py:class:`frozenset`


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
