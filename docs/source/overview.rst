Overview of hydra-zen
=====================

hydra-zen provides simple, Hydra-compatible tools that enable Python-centric workflows for designing, configuring, and running large-scale projects, such as machine learning experiments.
These functions can be broken down into two categories:

1. Functions like :func:`~hydra_zen.builds` and :func:`~hydra_zen.just` make it simple to dynamically :doc:`generate and compose structured configurations </structured_configs>` of your code.
2. :func:`~hydra_zen.experimental.hydra_run` and :func:`~hydra_zen.experimental.hydra_multirun` make it possible to :doc:`launch Hydra jobs </experimental>` and access their results from within a Python session instead of from the commandline. This means that Hydra's powerful plugins – `parameter sweepers <https://hydra.cc/docs/next/plugins/ax_sweeper>`_ and various `job-launchers <https://hydra.cc/docs/next/plugins/submitit_launcher>`_ – can be leveraged within a Jupyter notebook or any other Python process.

Libraries like `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ help to eliminate boilerplate
code – for-loops and other control-flow logic – associated with training and testing a neural network;
Hydra with hydra-zen follows suite and eliminates the code you would write to configure, orchestrate, and organize the results of your various experiments.



What is Hydra?
--------------

`Hydra <https://github.com/facebookresearch/hydra>`_ is a framework for elegantly configuring applications, such as a web app, and launching them from the commandline.
It provides a specification for writing rich, yaml-serializable configurations for your code.
It also makes simple the process of configuring and launching instances of your application from the commandline using various launchers and parameter-sweeper plugins.

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

hydra-zen is fully compatible with Hydra.
It provides tools that:

**Make it easy and ergonomic to dynamically generate structured configurations of your code as well as code from third-party libraries.**

   This replaces the cumbersome process of manually-defining configs or relying on statically-generated code.
   While Hydra excels at configuring and launching traditional software applications, hydra-zen is designed with data science and machine learning practitioners in mind: users whose code may have *many* intricately-configurable components and where manually-defined configs become a :ref:`hotspot for repetitious and error-prone workflows <DRY>`

**Enable Hydra-based work flows that are Python-centric rather than being CLI-centric.**

   hydra-zen targets the user whose end goal is not necessarily to launch a configurable app from the command line, but to run experiments/analysis in a way that is repeatable, scalable, and self-documenting.
