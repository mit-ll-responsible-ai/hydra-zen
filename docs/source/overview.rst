Overview of hydra-zen
=====================

hydra-zen provides simple, Hydra-compatible tools that enable Python-centric workflows for designing, configuring, and running large-scale projects, such as machine learning experiments.

What is Hydra?
--------------

`Hydra <https://github.com/facebookresearch/hydra>`_ is a framework for elegantly configuring applications, such as a web app, and launching them from the commandline.
It provides a specification for writing rich, yaml-serializable configurations for your code.
It also makes simple the process of configuring and launching instances of your application from the commandline using various launchers and parameter-sweeper plugins.

For example, the following code snippets demonstrates how one can configure, design, and run a toy application using Hydra.

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

Hydra will also handle directory management so that logs and results from a given job are organized automatically.

Check out Hydra's `great documentation <https://hydra.cc/>`_ that explains its design and purpose; it has detailed tutorials as well. And give it a star `on Github <https://github.com/facebookresearch/hydra>`_ if you haven't already!


What is hydra-zen?
------------------

hydra-zen is fully compatible with Hydra; it removes the need to manually write yaml configuration files (or dataclass-based configurations) for large projects and it enables users
to run configured programs within Python program rather than through the commandline interface.


Its functions can be broken down into two categories:

1. Functions like :func:`~hydra_zen.builds` and :func:`~hydra_zen.just` make it simple to dynamically :doc:`generate and compose structured configurations </structured_configs>` of your code.
2. :func:`~hydra_zen.experimental.hydra_run` and :func:`~hydra_zen.experimental.hydra_multirun` make it possible to :doc:`launch Hydra jobs </experimental>` and access their results from within a Python session instead of from the commandline. This means that Hydra's powerful plugins – `parameter sweepers <https://hydra.cc/docs/next/plugins/ax_sweeper>`_ and various `job-launchers <https://hydra.cc/docs/next/plugins/submitit_launcher>`_ – can be leveraged within a Jupyter notebook or any other Python process.

For example, let's suppose that we want to configure and run the following function

.. code-block:: python

   # some code to configure and run
   def repeat_text(num: int, text: str) -> str:
       return text * num

Rather than manually write a yaml file or define a dataclass to configure this function, we can use :func:`~hydra_zen.builds`
to dynamically generate a "structured configuration" for this function for us.

.. code-block:: python

   from hydra_zen import builds

   # generates a dataclass object that can configure/run `repeat_text`
   Config = builds(repeat_text, populate_full_signature=True)

This structured configuration can be "instantiated" by Hydra, which means that `repeat_text` will be "built" using the specified configuration values;
this instantiation will recurse through

.. code-block:: pycon

   >>> from hydra_zen import instantiate
   >>> config_instance = Config(num=3, text="world ")
   >>> instantiate(config_instance)
   'world world world '

This may not seem like a big deal – especially in the scope of this trivial toy example – but consider applications for which there are many configurable components that must be accounted for;
:func:`~hydra_zen.builds` can eliminate :ref:`error-prone and cumbersome processes <DRY>` for manually creating such configurations via yaml files or dataclass definitions.
It is worthwhile to review all of the :ref:`functionality <Builds>` that :func:`~hydra_zen.builds` affords us.


hydra-zen also makes it easy for us to run this configured code from within a Python process rather than from a commandline interface.


.. code-block:: python

   >>> from hydra_zen import instantiate
   >>> from hydra_zen.experimental import hydra_run
   >>> job_out = hydra_run(
   ...     Config(num=2, text="hello"), task_function=instantiate,
   ... )
   >>> job_out.return_value
   'hellohello'

It is useful to see hydra-zen in more realistic settings.
:ref:`Consider this example <Lightning>` of hydra-zen being used to configure a PyTorch-Lightning application.
`Here is a larger-scale example <https://github.com/mit-ll-responsible-ai/hydra-zen-examples>`_ of hydra-zen being used to configure
training code for a neural network image classifier.

Please continue through the rest of hydra-zen's documentation to learn more about Hydra and the features that hydra-zen brings
to the table.