.. meta::
   :description: hydra-zen eliminates the repetitive patterns involved in designing a Hydra-based project by providing users with functions for dynamically generating configurations for their project.

.. _dry:

========================================================================
Don't Repeat Yourself: Keeping DRY with Dynamically-Generated Configs 🌞
========================================================================

**DRY – Don't Repeat Yourself** – is a `principle of software development <https://en.
wikipedia.org/wiki/Don%27t_repeat_yourself>`_ that cautions against repetitive software 
patterns and workflows. One of the major benefits of using hydra-zen's 
:ref:`config-creation functions <create-config>` is that they help us abide by the DRY 
principle. On the other hand, manually writing configs for  your code will often leave 
you soaking **WET (Writing Everything Twice)**.

To develop an intuition for these claims, let's suppose that we want to configure the following class.

.. code-block:: python
   :caption: Contents of ``vision/model.py``

   from typing import Tuple

   # This is a class that we want to configure
   class DNN:
       def __init__(
           self,
           input_size: int,
           output_size: int,
           layer_widths: Tuple[int, ...] = (5, 10, 5),
           device: str = "cpu",
       ):
           pass

We'll compare the processes of configuring :class:`DNN` using a YAML-file, using a hand-written dataclass, and using :func:`hydra_zen.builds`.

Statically-Defined Configs are WET
==================================

Manually writing a structured config for ``DNN`` entails hard-coding its current import-path as a string as well as explicitly mirroring its signature:


.. tab-set::

   .. tab-item:: dataclass-based config

      .. code-block:: python
         :caption: Manually configuring ``DNN`` with a dataclass
      
         from dataclasses import dataclass
      
         @dataclass
         class Builds_DNN:
             input_size: int
             output_size: int
             layer_widths: tuple[int, ...] = (5, 10, 5)
             device: str = "cpu"
             _target_: str = "vision.model.DNN"

   .. tab-item:: yaml-based config

      .. code-block:: yaml
         :caption: Manually configuring ``DNN`` with a YAML file
      
         _target_: vision.model.DNN
         input_size: ???
         output_size: ???
         layer_widths:
         - 5
         - 10
         - 5
         device: cpu


Doing this once isn't so cumbersome, but consider that any time we modify ``DNN`` by:

- changing its location (e.g. move it to ``vision.classifiers.DNN``)
- updating any of its parameters' names, default values, or annotations
- adding or removing a parameter to/from its signature

then we will need to mirror this change in our configuration as well. I hope you 
brought your towel, because things are getting WET 🌊.

Having to manually sync our configs with our code is not only tedious but it also 
creates a hot-spot for mistakes and bugs.


Generating Configs with `builds` Keeps Us DRY
=============================================

We can stay nice and DRY by dynamically generating our configurations with :func:`~hydra_zen.builds`.

.. code-block:: python
   :caption: Configuring ``DNN`` using :func:`~hydra_zen.builds`
   
   from hydra_zen import builds
   from vision.model import DNN

   ZenBuilds_DNN = builds(DNN, populate_full_signature=True)


Here we don't need to worry about repeating ourselves in order to keep our config in 
sync with our code: the config (complete with type annotations and default values) is 
*automatically* and *dynamically* generated for us at runtime! 🌞

Additionally, any configured parameters that we do manually specify via 
:func:`~hydra_zen.builds` will be :ref:`validated against the signature of 
<builds-validation>` :class:`DNN` as the config is being created. Thus typos and 
mistakes will be caught fast and early - before we even launch our app.
