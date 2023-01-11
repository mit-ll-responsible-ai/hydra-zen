.. _launch:


Python Interface for Launching Hydra Applications
=================================================

hydra-zen provides a function, :func:`hydra_zen.launch`, that is a python based interface for launching hydra jobs.  
This function enables:

- Fully launching Hydra `RUN` and `MULTIRUN` jobs in a python script or interactive environment
- Python based approach to overrides of configuration values
- Writing tests without using `subprocess`

The main disadvantage to using :func:`hydra_zen.launch` is that it does not support Hydra CLI and therefore does 
not automatically support being delivered as an application. 

Those who may may benefit in using :func:`hydra_zen.launch` over Hydra CLI include:

- Developers who have not finalized an API design and want to easily test and iterate in an interactive environment
- Researchers who need to experiment with many different configuration options, a task that is often more easily done in a python script
- Writing tests with :func:`hydra_zen.launch` will remove the need to execute Hydra CLI within a `subprocess`


Comparing Hydra CLI and :func:`hydra_zen.launch`
----------------------------------------------

First lets create a Hydra application:

.. code-block:: python
   :caption: Example Hydra Application

   from hydra_zen import store, zen

   @store(name="my_app")
   def task_fn(foo):
      print(foo)

   if __name__ == "__main__":
      store.add_to_hydra_store()
      zen(task_fn).hydra_main(config_path=None)

The application can be executed using Hydra CLI:

.. code-block:: bash

    $ python my_app.py +foo=1
    1

To multirun over multiple values of `foo`, simply create a
Hydra multirun run list of values:

.. code-block:: bash

   $ python main.py +foo=1,2,3 --multrun
   [2023-01-11 10:54:10,532][HYDRA] Launching 3 jobs locally
   [2023-01-11 10:54:10,532][HYDRA]        #0 : +foo=1
   1
   [2023-01-11 10:54:10,623][HYDRA]        #1 : +foo=2
   2
   [2023-01-11 10:54:10,720][HYDRA]        #2 : +foo=3
   3

The same tasks can be executed in a single script using :func:`hydra_zen.launch`:

.. code-block:: python
   :caption: Example Script for `launch`

   from hydra_zen import launch, make_config, zen

   TestConfig = make_config()


   @zen
   def task_fn(foo):
      print(foo)


   # run two Hydra jobs within one script
   job = launch(TestConfig, task_fn, overrides=["+foo=1"])
   # outputs:
   # 1

   mulritun_job = launch(TestConfig, task_fn, overrides=["+foo=1,2,3"], multirun=True)
   # outputs:
   # [2023-01-11 10:56:02,448][HYDRA] Launching 3 jobs locally
   # [2023-01-11 10:56:02,448][HYDRA]        #0 : +foo=1
   # 1
   # [2023-01-11 10:56:02,537][HYDRA]        #1 : +foo=2
   # 2
   # [2023-01-11 10:56:02,626][HYDRA]        #2 : +foo=3
   # 3



Additionally, :func:`hydra_zen.launch` supports dictionary overrides:

.. code-block:: python
   :caption: Example Script for `launch` with Dictionary Overrides

   from hydra_zen import hydra_list, launch, make_config, multirun, zen

   TestConfig = make_config()


   @zen
   def task_fn(foo):
      print(foo)


   # run two Hydra jobs within one script
   job = launch(TestConfig, task_fn, overrides={"+foo": 1})
   # outputs:
   # 1

   # define a multirun list using `hydra_zen.multirun`
   mulritun_job = launch(
      TestConfig, task_fn, overrides={"+foo": multirun([1, 2, 3])}, multirun=True
   )
   # outputs:
   # [2023-01-11 10:56:02,448][HYDRA] Launching 3 jobs locally
   # [2023-01-11 10:56:02,448][HYDRA]        #0 : +foo=1
   # 1
   # [2023-01-11 10:56:02,537][HYDRA]        #1 : +foo=2
   # 2
   # [2023-01-11 10:56:02,626][HYDRA]        #2 : +foo=3

   # define a standard Hydra list as a single parameter using `hydra_zen.hydra_list`
   mulritun_job = launch(TestConfig, task_fn, overrides={"+foo": hydra_list([1, 2, 3])})
   # outputs:
   # [1, 2, 3]


One clear benefit of :func:`hydra_zen.launch` is the ability to programmatically define the set of
multirun values, e.g., creating a list of random seeds to execute an application with.


  

