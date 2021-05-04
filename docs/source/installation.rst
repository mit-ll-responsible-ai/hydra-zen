Installation and Basic Usage
============================

Installing hydra-zen
--------------------

hydra-zen is lightweight: its only dependencies are ``hydra-core`` and ``typing-extensions``.
It is available for installation via pip:

.. code-block:: shell

  pip install hydra-zen

We strive to maintain as wide a window of backwards compatibility Hydra (and omegaconf) as possible.


Basic Usage
-----------

The following is a simple example in which we use hydra-zen to create a structured config for a function.
We will then launch multiple jobs that each run the function with a different set of configuration values.

.. code-block:: python

   from hydra_zen import builds, instantiate, to_yaml
   from hydra_zen.experimental import hydra_multirun

   # some code to configure and run
   def repeat_text(num: int, text: str) -> str:
       return text * num

.. code-block:: pycon

   # Create structured config without providing default values.
   # This is a dataclass object.
   >>> Config = builds(repeat_text, populate_full_signature=True)

   # Instantiating the config builds `repeat_text`
   # I.e. it calls `repeat_text(num=2, text="hi")`
   >>> instantiate(Config(num=2, text="hi"))
   'hihi'

   # `builds` retains the type hints from the function, which
   # Hydra can use to provide runtime type validation
   >>> instantiate(Config(num=2.5, text="hi"))
   ValidationError: Value '2.5' could not be converted to Integer
       full_key: num
       object_type=Builds_repeat_text

   # run multiple jobs over combination of parameter values
   >>> jobs, = hydra_multirun(
   ...     Config,
   ...     task_function=instantiate,
   ...     overrides=["num=1,2,3", "text=hi,bye"],
   ... )
   [2021-04-30 15:31:02,688][HYDRA] Launching 6 jobs locally
   [2021-04-30 15:31:02,689][HYDRA] 	#0 : text=hi num=1
   [2021-04-30 15:31:02,764][HYDRA] 	#1 : text=hi num=2
   [2021-04-30 15:31:02,838][HYDRA] 	#2 : text=hi num=3
   [2021-04-30 15:31:02,915][HYDRA] 	#3 : text=bye num=1
   [2021-04-30 15:31:02,988][HYDRA] 	#4 : text=bye num=2
   [2021-04-30 15:31:03,063][HYDRA] 	#5 : text=bye num=3

   # access results of jobs
   >>> [j.return_value for j in jobs]
   ['hi', 'hihi', 'hihihi', 'bye', 'byebye', 'byebyebye']

   # get the config of the first run as a yaml
   >>> print(to_yaml(jobs[0].cfg))
   _target_: my_lib.repeat_text
   _recursive_: true
   _convert_: none
   text: hi
   num: 1
