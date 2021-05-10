.. hydra-zen documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hydra-zen's documentation!
=====================================

hydra-zen helps you configure your project using the power of `Hydra <https://github.com/facebookresearch/hydra>`_, while enjoying the `Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_!

hydra-zen provides simple, Hydra-compatible tools that enable Python-centric workflows for designing, configuring, and running large-scale projects, such as machine learning experiments.
It helps to eliminate the boilerplate code you would write to configure, orchestrate, and organize the results of your various experiments.

hydra-zen offers:
  - Functions for automatically and dynamically generating `structured configs <https://hydra.cc/docs/next/tutorials/structured_config/schema/>`_ that can be used to fully or partially instantiate objects in your application.
  - The ability to launch Hydra jobs, complete with parameter sweeps and multi-run configurations, from within a notebook or any other Python environment.
  - Incisive type annotations that provide enriched context about your project's configurations to IDEs, type checkers, and other tooling.
  - Runtime validation of configurations to catch mistakes before your application launches.
  - Equal support for both object-oriented libraries (e.g., ``torch.nn``) and functional ones (e.g., ``jax`` and ``numpy``).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   overview
   pytorch_lightning_example
   structured_configs
   experimental

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
