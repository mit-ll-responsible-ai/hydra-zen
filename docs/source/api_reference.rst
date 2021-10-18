#############
API Reference
#############

*************************
Launching Jobs with Hydra
*************************

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   launch

.. currentmodule:: hydra_zen.experimental

.. autosummary::
   :toctree: generated/
   
   hydra_run
   hydra_multirun


********************************************
Creating and Working with Structured Configs
********************************************

hydra-zen provides us with some simple but powerful tools for creating and working with structured configs.


Creating Structured Configs
***************************
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

hydra-zen offers support for customized runtime type-checking via various third-party libraries.

.. currentmodule:: hydra_zen.third_party

.. autosummary::
   :toctree: generated/

   beartype.validates_with_beartype
   pydantic.validates_with_pydantic
