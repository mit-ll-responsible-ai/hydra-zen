#############
API Reference
#############

*********************************
Creating and Working with Configs
*********************************

hydra-zen provides us with some simple but powerful tools for creating and working with structured configs.



.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   make_config
   builds
   instantiate
   make_custom_builds_fn
   get_target
   just
   hydrated_dataclass
   ZenField


*************************
Launching Jobs with Hydra
*************************

.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   launch

*********************
Third-Party Utilities
*********************


Runtime Type-Validation
***********************

.. currentmodule:: hydra_zen.third_party

.. autosummary::
   :toctree: generated/

   beartype.validates_with_beartype
   pydantic.validates_with_pydantic
