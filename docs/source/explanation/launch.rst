.. _launch:


Python Interface for Launching Hydra Applications
=================================================

hydra-zen provides a function, :func:`hydra_zen.launch`, that is a Python-based interface for launching Hydra jobs.
This function enables:

- Fully launching Hydra `RUN` and `MULTIRUN` jobs from a Python script or from within an interactive Python session.
- A programmatic approach to overrides of configuration values (i.e. using code to construct a job's overrides).
- Running / testing Hydra jobs without relying on `subprocess.run`.



Comparing Hydra CLI and :func:`hydra_zen.launch`
------------------------------------------------

First lets create a Hydra application:

.. code-block:: python
   :caption: Example Hydra Application (`my_app.py`)

   from hydra_zen import store, zen

   @store(name="my_app")
   def task_fn(foo):
       print(foo)

   if __name__ == "__main__":
      store.add_to_hydra_store()
      zen(task_fn).hydra_main(config_path=None, version_base="1.2")

The application can be executed using Hydra CLI:

.. code-block:: bash

    $ python my_app.py +foo=1
    1

To multirun over multiple values of `foo`, simply create a
Hydra multirun run list of values:

.. code-block:: bash

   $ python my_app.py +foo=1,2,3 --multirun
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

   # Just an empty config to demonstrate overrides
   TestConfig = make_config()


   def task_fn(foo):
       print(foo)

   # Use of `zen` to make a Hydra-compatible task function.
   zen_fn = zen(task_fn)

   # run two Hydra jobs within one script
   job = launch(TestConfig, zen_fn, overrides=["+foo=1"], version_base="1.2")
   # outputs:
   # 1

   multirun_job = launch(TestConfig, zen_fn, overrides=["+foo=1,2,3"], multirun=True, version_base="1.2")
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

   # Just an empty config to demonstrate overrides
   TestConfig = make_config()


   def task_fn(foo):
      print(foo)

   # Use of `zen` to make a Hydra-compatible task function.
   zen_fn = zen(task_fn)

   # Run two Hydra jobs within one script.
   job = launch(TestConfig, zen_fn, overrides={"+foo": 1}, version_base="1.2")
   # outputs:
   # 1

   # define a multirun list using `hydra_zen.multirun`
   multirun_job = launch(
      TestConfig, zen_fn, overrides={"+foo": multirun([1, 2, 3])}, multirun=True, version_base="1.2"
   )
   # outputs:
   # [2023-01-11 10:56:02,448][HYDRA] Launching 3 jobs locally
   # [2023-01-11 10:56:02,448][HYDRA]        #0 : +foo=1
   # 1
   # [2023-01-11 10:56:02,537][HYDRA]        #1 : +foo=2
   # 2
   # [2023-01-11 10:56:02,626][HYDRA]        #2 : +foo=3

   # define a standard Hydra list as a single parameter using `hydra_zen.hydra_list`
   multirun_job = launch(TestConfig, zen_fn, overrides={"+foo": hydra_list([1, 2, 3])}, version_base="1.2")
   # outputs:
   # [1, 2, 3]


One clear benefit of :func:`hydra_zen.launch` is the ability to programmatically define the set of
multi-run values, e.g., creating a list of random seeds to execute an application with.


  

