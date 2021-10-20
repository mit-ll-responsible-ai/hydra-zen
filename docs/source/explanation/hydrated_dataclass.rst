Combining Static and Dynamic Configurations with `@hydrated_dataclass`
======================================================================

hydra-zen provides a decorator, `hydrated_dataclass`, which is similar to the standard `@dataclass` but can be used to auto-populate Hydra-specific parameters;
it also exposes other features that are available in `builds`.

.. code:: python

   from hydra_zen import hydrated_dataclass

   from torch.optim import Adam

   @hydrated_dataclass(target=Adam, zen_partial=True, frozen=True)
   class BuildsAdam:
       lr: float = 0.01
       momentum: float = 0.9

   BuildsAdam(lr="a string")  # static type-checker flags as invalid (invalid type)

   conf = BuildsAdam()
   conf.lr = 10.0  # static type-checker flags as invalid (mutating "frozen" dataclass)


This has the benefit of making certain pertinent information (e.g. the dataclass' fields and that it is frozen) available to static type checkers, while still dynamically populating the resulting dataclass with Hydra-specific fields (e.g. ``_target_`` and ``_partial_target_``) and providing the same runtime validation as `builds`.

Note that the ``@hydrated_dataclass`` decorator uses a `recently proposed <https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md>`_ mechanism for enabling static tools to "recognize" third-party dataclass decorators like this one.
Presently, the above static inspection is only supported by pyright, but other type-checkers will likely add support for this soon.
