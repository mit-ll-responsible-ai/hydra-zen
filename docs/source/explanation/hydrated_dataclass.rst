Combining Statically-Defined and Dynamically-Generated Configurations
=====================================================================

hydra-zen provides a decorator, :func:`~hydra_zen.hydrated_dataclass`, which is similar 
to :py:func:`dataclasses.dataclass`, but can be used to auto-populate Hydra-specific 
parameters; it also exposes other features that are available in 
:func:`~hydra_zen.builds`.

E.g. in the following codeblock, we will use ``@hydrated_dataclass`` to create a frozen
(i.e. immutable) config, which is designed to partially configure the class 
``torch.optim.Adam``.

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


.. code:: pycon

   >>> from hydra_zen import instantiate
   >>> instantiate(BuildsAdam)
   functools.partial(<class Adam>, lr=0.01, momentum=0.9)

This has the benefit of making certain pertinent information (e.g. the dataclass' 
fields and that it is frozen) available to static type checkers, while still 
dynamically populating the resulting dataclass with Hydra-specific fields (e.g. 
``_target_``) and providing the same runtime validation capabilities as 
:func:`~hydra_zen.builds` – e.g. the following code will raise an error during the
creation of ``BuildsAdam`` because the field name ``momentum`` was misspelled.

.. code:: python

   # @hydrated_dataclass will catch the misspelled parameter name
   # and will raise a TypeError

   @hydrated_dataclass(target=Adam)
   class BuildsAdam:
       momtum: float = 0.9  # <- typo causes TypeError upon constructing config

This means that our config is validated *upon construction*: we will identify this 
error before we launch our Hydra job.

Note that the ``@hydrated_dataclass`` decorator uses a `recently proposed <https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md>`_ mechanism for 
enabling static tools to "recognize" third-party dataclass decorators like this one.
Presently, the above static inspection is only supported by pyright, but other 
type-checkers will likely add support for this soon.
