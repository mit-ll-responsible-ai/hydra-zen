.. meta::
   :description: hydra-zen provides a dataclass-like decorator for users to combine statically-defined and dynamically-generated fields in a configuration for a Hydra application.


Combining Statically-Defined and Dynamically-Generated Configurations
=====================================================================

hydra-zen provides a decorator, :func:`~hydra_zen.hydrated_dataclass`, which is similar 
to :py:func:`dataclasses.dataclass`. It can be used to dynamically auto-populate configuration parameters, à la :func:`~hydra_zen.builds`. However, it also enables users to define a config such that its attributes are statically available to various tools, like type-checkers and IDEs.

E.g. in the following codeblock, we will use ``@hydrated_dataclass`` to create a frozen
(i.e. immutable) config, which is designed to partially configure the class 
``torch.optim.Adam``. Here, static tooling can "see" the types associated with the
configured fields, and flag bad inputs in our code. That this config is immutable is also salient to static analysis.

.. code-block:: python

   from hydra_zen import hydrated_dataclass 

   from torch.optim import Adam

   @hydrated_dataclass(target=Adam, zen_partial=True, frozen=True)
   class BuildsAdam:
       lr: float = 0.01
       momentum: float = 0.9

   # static type-checker flags as invalid (invalid type)
   BuildsAdam(lr="a string")  # type: ignore

   conf = BuildsAdam()
   # static type-checker flags as invalid (mutating "frozen" dataclass)
   conf.lr = 10.0  # type: ignore


.. code-block:: pycon

   >>> from hydra_zen import instantiate
   >>> instantiate(BuildsAdam)
   functools.partial(<class Adam>, lr=0.01, momentum=0.9)

Note that we did not need to specify Hydra-specific fields like ``_target_``: the 
decorator handled this for us. Furthermore, we also benefit from the additional runtime validation capabilities that :func:`~hydra_zen.builds` provides; e.g. the following 
code will raise an error during the creation of ``BuildsAdam`` because the field name 
``momentum`` was misspelled.

.. code-block:: python

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
