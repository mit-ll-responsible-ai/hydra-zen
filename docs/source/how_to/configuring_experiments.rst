.. meta::
   :description: Configuring and maintaining multiple experiment configurations.


==================================
Configurating Multiple Experiments
==================================

hydra-zen provides simple and efficient tools to generate multiple experiment configurations. You can easily create new configuration 
instances that inherit other configurations while only specifying changes to different parameter values and defaults. 
This approach can save a significant amount of time and reduce the potential for errors when setting up large-scale experiments.
In this documentation, we will go through the process of using hydra-zen to generate multiple experiment configurations,
and provide examples of how to use the module to streamline your workflow.

We will demonstrate how one can implement the solution shown in `Configurating Experiments <https://hydra.cc/docs/patterns/configuring_experiments/>`_.
First lets build the default configurations and task function. To build and store configurations in the
`Hydra ConfigStore <https://hydra.cc/docs/tutorials/structured_config/config_store/>`_ we will take advantage of :func:`~hydra_zen.store`. 
Here we add two configurations to both the `db` and `server` groups.


.. code-block:: python
    :caption: 1: Default Configurations
   
    from dataclasses import dataclass

    from hydra_zen import MISSING, make_config, store, zen

    # First create stores for each group
    db_store = store(group="db")
    server_store = store(group="server")

    # add to named configurations for the `db` group
    @db_store(name="mysql")
    @dataclass
    class MySQL:
        name: str = "mysql"


    @db_store(name="sqlite")
    @dataclass
    class SQLITE:
        name: str = "sqlite"


    # add to named configurations for the `server` group
    @server_store(name="apache")
    @dataclass
    class Apache:
        name: str = "apache"
        port: int = 80


    @server_store(name="nginx")
    @dataclass
    class NGINX:
        name: str = "nginx"
        port: int = 80


    # utilize `make_config` to easily create configurations with
    # hydra defaults
    Config = make_config(
        hydra_defaults=["_self_", {"db": "sqlite"}, {"server": "apache"}],
        db=MISSING,
        server=MISSING,
    )

    # put main configuration in the store
    store(
        Config,
        name="config",
    )

Next define the task function for our application:

.. code-block:: python
    :caption: 2: Python Application `my_app.py`

    from hydra_zen import MISSING, make_config, store, to_yaml, zen

    # ... load configurations

    def task(db, server):
        print("db:")
        print(to_yaml(db))

        print("server:")
        print(to_yaml(server))


    if __name__ == "__main__":
        store.add_to_hydra_store()
        zen(task).hydra_main(config_path=None, config_name="config", version_base="1.2")


The application can then be executed using:

.. code-block:: bash

    $ python my_app.py
    db:
    name: mysql
    
    server:
    name: apache
    port: 80


Our objective is to create experiment configurations that override the default using

.. code-block:: bash

    $ python my_app.py +experiment=fast_mode


To do this we implement new experiment configurations that:

- Are global configurations using `package="_global_"` and inherting from the default `Config`
- Override defaults configuration values using absolute paths for `/db` and `/server`
- Override parameter values
  
.. code-block:: python
    :caption: 3: Experiment Configurations

    # the experiment configs:
    # - must inherit from `Config` 
    # - must set `package="_global_"`

    experiment_store(
        make_config(
            hydra_defaults=["_self_", {"override /db": "sqlite"}],
            server=dict(port=8080),
            bases=(Config,),
        ),
        name="aplite",
        package="_global_",
    )


    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /db": "sqlite"},
                {"override /server": "nginx"},
            ],
            server=dict(port=8080),
            bases=(Config,)
        ),
        name="nglite",
        package="_global_"
    )

Experiments can then be run from command line by prefixing the experiment choice with a `+` since the
experiment config group is an addition and not an override. Here are couple examples:

.. code-block:: bash

    $ python my_app.py  +experiment=aplite
    db:
    name: sqlite

    server:
    name: apache
    port: 8080

    $ python my_app.py --multirun +experiment=aplite,nglite
    [2023-01-17 10:45:25,609][HYDRA] Launching 2 jobs locally
    [2023-01-17 10:45:25,609][HYDRA]        #0 : +experiment=aplite
    db:
    name: sqlite

    server:
    name: apache
    port: 8080

    [2023-01-17 10:45:25,713][HYDRA]        #1 : +experiment=nglite
    db:
    name: sqlite

    server:
    name: nginx
    port: 8080


Alternative Approaches
======================

In order to keep the concepts simple we focused on the use of :py:func:`dataclasses.dataclass`. 
We could easily utilize :func:`~hydra_zen.make_config` as shown below

.. code-block:: python

    db_store(make_config(name="mysql"), name="mysql")
    ...

    server_store(make_config(name="apache", port=80), name="apache")
    ...

Another neat trick that :func:`~hydra_zen.store` provides, which is a bit subtle, is that you can avoid having 
to specify name twice by telling the store how to infer the store-entry's name from the config:

.. code-block:: python

    # will infer store-entry name from .name attr of config
    auto_name_store = store(name=lambda cfg: cfg.name)

    # First create stores for each group
    db_store = auto_name_store(group="db")
    server_store = auto_name_store(group="server")
    experiment_store = store(group="experiment")


    # add to named configurations for the `db` group
    db_store(make_config(name="mysql"))
    db_store(make_config(name="sqlite"))

    # add to named configurations for the `server` group
    server_store(make_config(name="apache", port=80))
    server_store(make_config(name="nginx", port=80))