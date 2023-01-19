.. meta::
   :description: Configuring Hydra.


===================
Configurating Hydra
===================

To configure Hydra using hydra-zen we must update the stored configuration using :class:`hydra.conf.HydraConf`.
Simply modify this :py:func:`dataclasses.dataclass` and overwrite the Hydra configuration stored under `group="hydra"`
and `name="config"`. Here is an example of using Hydra version `1.2` and setting the `chdir` flag to 
for experiments:


.. code-block:: python

    import os

    from hydra_zen import zen, store, make_config
    from hydra.conf import HydraConf, JobConf


    store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")

    store(make_config(), name="config")


    def task():
        print(os.getcwd())


    if __name__ == "__main__":
        # notice the `overwite_ok` flag
        store.add_to_hydra_store(overwrite_ok=True)
        zen(task).hydra_main(config_path=None, config_name="config", version_base="1.2")

Take note that because we must replace the current Hydra configuration in the Hydra `ConfigStore` we
must set the flag `overwrite_ok=True`.

