hydra\_zen.store
================


`hydra_zen.store` is a pre-instantiated, globally-available instance of :class:`hydra_zen.ZenStore`.

It is instantiated as:

.. code-block:: python

   from hydra_zen import ZenStore

   store = ZenStore(
       name="zen_store",
       deferred_to_config=True,
       deferred_hydra_store=True,
   )

