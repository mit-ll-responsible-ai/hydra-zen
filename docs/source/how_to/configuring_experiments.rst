.. meta::
   :description: Configuring and maintaining multiple experiment configurations.


******************************
Configure Multiple Experiments
******************************

This guide demonstrates how to elegantly maintain multiple configurations for an experiment, so that each experiment's config need only specify changes to some "master" config [1]_.  


Suppose that we want to benchmark an application that is configured using various pairings of databases (MySQL and SQLite) and servers (Apache and NGINX).
And, suppose that there are particular database-server configurations that we want to frequently experiment with. In this How-To we will:

1. Create the basic server and database configurations for our application and add them to a config store.
2. Define a simple task function that is executed when we configure and run our application from the CLI.
3. Create specialized configurations for particular experiments that we want to run, so that it is trivial to run each experiment "by name" from the CLI.
4. Run various configurations of our experiment.


Configuring and Running our Basic Application
=============================================

First, we well create configs for our two servers and our two databases and we will add them to :class:`hydra-zen's config store <hydra_zen.ZenStore>`.

.. code-block:: python
   :caption: Contents of `my_app.py`

   from dataclasses import dataclass
   
   from hydra_zen import store, zen, builds

   # 1. Creating and storing basic configs

   @dataclass
   class Server:
       name: str
       port: int
   
   
   @dataclass
   class Database:
       name: str
   
   # For convenience:
   # Tell the store to automatically infer the entry-name from 
   # the config's `.name` attribute.
   auto_name_store = store(name=lambda cfg: cfg.name)
   
   # Pre-set the group for store's db entries
   db_store = auto_name_store(group="db")
   
   db_store(Database(name="mysql"))
   db_store(Database(name="sqlite"))
   
   # Pre-set the group for store's server entries
   server_store = auto_name_store(group="server")
   
   server_store(Server(name="apache", port=80))
   server_store(Server(name="nginx", port=80))


   # 2. Defining our app's task function and the top-level config
   def task(db: Database, server: Server):
       from hydra_zen import to_yaml
   
       print(f"db:\n{to_yaml(db)}")
       print(f"server:\n{to_yaml(server)}")


   # The 'top-level' config for our app w/ a specified default
   # database and server
   Config = builds(
       task,
       populate_full_signature=True,
       hydra_defaults=["_self_", {"db": "mysql"}, {"server": "apache"}],
   )
   
   store(Config, name="config")

   if __name__ == "__main__":
       store.add_to_hydra_store()
       zen(task).hydra_main(config_path=None, config_name="config", version_base="1.2")


The application can then be executed using:

.. code-block:: console
   :caption: Running our application using the default config.

   $ python my_app.py
   db:
   name: mysql
    
   server:
   name: apache
   port: 80



Creating Configurations for Particular "Experiments"
====================================================

Suppose that we frequently want to run our application using the following two configurations (which we will refer to as `aplite` and `nglite`, respectively)

.. code-block:: console
   :caption: Manually running the so-called `aplite` configuration

   $ python my_app.py db=sqlite server.port=8080
   db:
   name: sqlite

   server:
   name: apache
   port: 8080


.. code-block:: console
   :caption: Manually running the so-called `nglite` configuration
   
   $ python my_app.py db=sqlite server=nginx server.port=8080                                              
   db:
   name: sqlite

   server:
   name: nginx
   port: 8080


Our objective is to be able run these experiments more concisely, as:

.. code-block:: console

    $ python my_app.py +experiment=<aglite or nginx>


To do this we implement new experiment configurations that:

- Inherit from `Config` – the default config for our app – so that Hydra will be able to compose it with each experiment's config.
- Are stored under the `_global_` `package <https://hydra.cc/docs/advanced/overriding_packages/>`_, so that they are used to replace our top-level config, and under a group called "experiment", which will determine how we reference them from the CLI.
- Override defaults configuration values using absolute paths for `/db` and `/server`. Using absolute paths is necessary given that we are not leveraging `Hydra's config search path logic <https://hydra.cc/docs/advanced/search_path/>`_ (which is typically reserved for yaml-based configs).
- Override particular parameter values (i.e., the configured server port)
  
.. code-block:: python
    :caption: 3: Adding experiment configs (an addition to `my_app.py`)

    # add the following before the __main__ clause of `my_app.py`

    from hydra_zen import make_config

    # the experiment configs:
    # - must be stored under the _global_ package
    # - must inherit from `Config` 
    experiment_store = store(group="experiment",  package="_global_")

    # equivalent to `python my_app.py db=sqlite server.port=8080`
    experiment_store(
        make_config(
            hydra_defaults=["_self_", {"override /db": "sqlite"}],
            server=dict(port=8080),
            bases=(Config,),
        ),
        name="aplite",
    )


    # equivalent to: `python my_app.py db=sqlite server=nginx server.port=8080`
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
    )

Now the configuration for either "experiment" can be specified by-name from the CLI. Because the `experiment` group is not present in the original config, we must specify the group with a `+` prefix: `+experiment`. Here are examples of ways that we can run our experiments from the CLI:

.. tab-set::

   .. tab-item:: aplit

      .. code-block:: console
         :caption: 4 Running the `aplite` experiment

         $ python my_app.py +experiment=aplite
         db:
         name: sqlite
     
         server:
         name: apache
         port: 8080


   .. tab-item:: nglite

      .. code-block:: console
         :caption: 4 Running the `nglite` experiment

         $ python my_app.py +experiment=nglite
         db:
         name: sqlite
         
         server:
         name: nginx
         port: 8080

   .. tab-item:: multi-run

      .. code-block:: console
         :caption: 4 Performing a multi-run over experiments
      
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


   .. tab-item:: multi-run (via glob)

      .. code-block:: console
         :caption: 4 Running all experiments using the `glob <https://hydra.cc/docs/advanced/override_grammar/extended/#glob-choice-sweep>`_ syntax

         $ python my_app.py --multirun '+experiment=glob(*)'
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


Footnotes
=========
.. [1] This closely mirrors Hydra's  `Configuring Experiments <https://hydra.cc/docs/patterns/configuring_experiments/>`_ guide, which describes a YAML-based solution to the same problem. In contrast to this, we emphasis a dataclass-based approach that leverages hydra-zen's enhanced functionality.
