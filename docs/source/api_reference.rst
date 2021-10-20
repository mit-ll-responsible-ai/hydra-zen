#############
API Reference
#############

*************************
Launching Jobs with Hydra
*************************

hydra-zen provides users the ability to launch a Hydra job via a 
Python function instead of from a commandline interface.

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   launch

.. currentmodule:: hydra_zen.experimental

.. autosummary::
   :toctree: generated/
   
   hydra_run
   hydra_multirun


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


*********************
Third-Party Utilities
*********************

.. _data-val:

Runtime Data Validation
***********************

Although Hydra provides some runtime type-checking functionality, it only supports `a limited set of types and annotations <https://omegaconf.readthedocs.io/en/latest/structured_config.html#simple-types>`_. hydra-zen offers support for more robust runtime type-checking capabilities via various third-party libraries.

.. currentmodule:: hydra_zen.third_party

.. autosummary::
   :toctree: generated/

   beartype.validates_with_beartype
   pydantic.validates_with_pydantic
