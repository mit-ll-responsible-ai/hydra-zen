Overview of hydra-zen
=====================

hydra-zen provides simple, Hydra-compatible tools that enable Python-centric workflows for designing, configuring, and running large-scale projects, such as machine learning experiments.
These functions can be broken down into two categories:

1. Functions like :func:`~hydra_zen.builds` and :func:`~hydra_zen.just` make it simple to dynamically :doc:`generate and compose structured configurations </structured_configs>` of your code.
2. :func:`~hydra_zen.experimental.hydra_run` and :func:`~hydra_zen.experimental.hydra_multirun` make it possible to launch Hydra jobs and access their results from within a Python session instead of from the commandline. This means that Hydra's powerful plugins – parameter sweepers and various job-launchers – can be leveraged within a Jupyter notebook or any other Python process.


What is Hydra?
--------------

`Hydra <https://github.com/facebookresearch/hydra>`_ is a framework for elegantly configuring applications, such as a web app, and launching them from the commandline. It provides a specification for writing rich, yaml-serializable configurations for your code, and makes it simple to configure and launch instances of your application from the commandline using various launchers and parameter-sweeper plugins.

+----------------------------+------------------------------------------+-------------------------------------------------+
| Configuration (yaml)       | Application                              | Launch with Overrides                           |
+============================+==========================================+=================================================+
| .. code:: yaml             | .. code:: python                         | .. code:: shell                                 |
|                            |                                          |                                                 |
|    db:                     |    import hydra                          |    $ python my_app.py db.port=8888              |
|     type: postgres         |    from omegaconf import OmegaConf       |    db:                                          |
|     host: localhost        |                                          |      type: postgres                             |
|     port: 5342             |    @hydra.main(config_name="config")     |      host: localhost                            |
|                            |    def my_app(cfg) -> None:              |      port: 8888                                 |
|                            |        print(OmegaConf.to_yaml(cfg))     |                                                 |
|                            |                                          |                                                 |
|                            |    if __name__ == "__main__":            |                                                 |
|                            |        my_app()                          |                                                 |
|                            |                                          |                                                 |
+----------------------------+------------------------------------------+-------------------------------------------------+


Check out Hydra's `great documentation <https://hydra.cc/>`_ that explains its design and purpose; it has detailed tutorials as well. And give it a star `on Github <https://github.com/facebookresearch/hydra>`_ if you haven't already!



These configurations can be written in a yaml format, or in Python via `structured configs <https://hydra.cc/docs/next/tutorials/structured_config/intro>`_.
Structured configs are dataclasses whose type annotations (up to a limited assortment) can be leveraged by Hydra to provide runtime type checking for your configurations.


What is hydra-zen?
------------------

hydra-zen is fully compatible with Hydra; it provides tools that make it easy and ergonomic to generate rich structured configurations, and that enable work flows that never leave Python – all while enjoying the benefits and power of Hydra.

These tools are especially useful for users working in data science and machine learning, where the end-goal of your work is not to launch a configurable app from the command line, but to configure and run experiments/analysis in a way that is repeatable, scalable, and self-documenting.


A structured config that you produce using, e.g., :func:`~hydra_zen.builds` is

.. code:: python

   def my_func(a_number: int, text: str):
        pass


+----------------------------+----------------------------------------+-------------------------------------------------+
| Using a yaml file          | Using a structured config              | Using hydra-zen                                 |
+============================+========================================+=================================================+
| .. code:: yaml             | .. code:: python                       | .. code:: pycon                                 |
|                            |                                        |                                                 |
|    _target_: mylib.my_func |    @dataclass                          |    # creates equivalent dataclass               |
|    number: 1               |    class Builds_my_func:               |    >>> builds(my_func, a_number=1, text="hello")|
|    text: hello             |        _target_ : str = "mylib.my_func"|    types.Builds_my_func                         |
|                            |        a_number : int = 1              |                                                 |
|                            |        text : str = "hello"            |                                                 |
|                            |                                        |                                                 |
+----------------------------+----------------------------------------+-------------------------------------------------+



