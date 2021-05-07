*****************************************
Dynamically Generating Structured Configs
*****************************************


API Reference
=============

hydra-zen provides us with some simple but powerful tools for dynamically generating structured configs for our code.
This helps to keep the process of configuring complex applications simple and intuitive.


.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   builds
   just
   hydrated_dataclass


Basics of Structured Configs
============================

Hydra supports configurations that are written in a yaml format or in Python via `structured configs <https://hydra.cc/docs/next/tutorials/structured_config/intro>`_.
Structured configs are `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ whose type annotations (up to a limited assortment) can be leveraged by Hydra to provide runtime type checking for your configurations.

An important feature of these structured configs is that they too can be serialized to yaml files by Hydra.
This is critical, as it ensures that each job that is launched using Hydra is fully documented by – and can be reproduced from – a plain-text yaml configuration.

A `targeted configuration <https://hydra.cc/docs/next/advanced/instantiate_objects/overview>`_ is designed to instantiate / call an object (a class-object or a function) with particular values.
hydra-zen provides functions that are specifically designed to created targeted structured configs.
For example, suppose that we want to configure the following class

.. code:: python

   # contents of vision/model.py
   from typing import Tuple

   # this is a class that we want to configure
   class DNN:
       def __init__(
           self,
           input_size: int,
           output_size: int,
           layer_widths: Tuple[int, ...] = (5, 10, 5),
           device: str = "cpu",
       ):

           print(
               f"DNN(input_size={input_size}, output_size={output_size}, layer_widths={layer_widths}, device={device})"
           )

and that we want the default values in our configuration to reflect the values specified in the class' signature.

Creating Hydra-Compatible Configurations
****************************************

The following table compares the two standard approaches for configuring ``DNN`` – using a manually-written yaml file or a manually-defined structured config – along with hydra-zen's approach of **dynamically generating** a structured config.

+-------------------------------+---------------------------------------------------+------------------------------------------------------+
| (Hydra) Using a yaml file     | (Hydra) Using a structured config                 | (hydra-zen) Using `builds`                           |
+===============================+===================================================+======================================================+
| .. code:: yaml                | .. code:: python                                  | .. code:: python                                     |
|                               |                                                   |                                                      |
|    _target_: vision.model.DNN |    from omegaconf import MISSING                  |    >>> from vision.model import DNN                  |
|    input_size: ???            |    from dataclasses import dataclass              |    >>> from hydra_zen import builds                  |
|    output_size: ???           |                                                   |    >>> builds(target=DNN,                            |
|    layer_widths:              |    @dataclass                                     |    ...        populate_full_signature=True,          |
|    - 5                        |    class Builds_DNN:                              |    ...        )                                      |
|    - 10                       |       _target_: str = "vision.model.DNN"          |    types.Builds_DNN                                  |
|    - 5                        |       input_size: int = MISSING                   |                                                      |
|    device: cpu                |       output_size: int = MISSING                  |                                                      |
|                               |       layer_widths: Tuple[int, ...] = (5, 10, 5), |                                                      |
|                               |       device: str = "cpu"                         |                                                      |
+-------------------------------+---------------------------------------------------+------------------------------------------------------+

Note that the result of ``builds(DNN, populate_full_signature=True)`` is *identical* to the manually-defined dataclass ``Builds_DNN``:
`builds` returns a dataclass object with parameters – along with their default values and type annotations – that are auto-populated based on the signature of ``DNN``.

.. code:: python

   >>> import inspect
   >>> from dataclasses import is_dataclass
   >>> Conf = builds(DNN, populate_full_signature=True)
   >>> is_dataclass(Conf)
   True
   >>> inspect.signature(Conf)
   <Signature (input_size: int, output_size: int, layer_widths: Tuple[int, ...] = (5, 10, 5), device: str = 'cpu') -> None>

As stated earlier, each of these targeted configurations can be used to instantiate ``DNN`` via Hydra:

.. code:: python

   # instantiating the targeted config will "build" an instance of `DNN`
   >>> from hydra_zen import instantiate  # annotated alias of `hydra.utils.instantiate`
   >>> Conf = builds(DNN, populate_full_signature=True)
   >>> instantiate(Conf(input_size=2, output_size=10, device="cuda:0"))
   DNN(input_size=2, output_size=10, layer_widths=[5, 10, 5], device=cuda:0)

And this dynamically generated configuration can still be serialized to a yaml file by Hydra:

.. code:: python

   >>> from hydra_zen import to_yaml  # alias of `omegaconf.OmegaConf.to_yaml`
   >>> print(to_yaml(Conf(input_size=2, output_size=10, device="cuda:0")))
   _target_: vision.model.DNN
   _recursive_: true
   _convert_: none
   input_size: 2
   output_size: 10
   layer_widths:
   - 5
   - 10
   - 5
   device: cuda:0


Hydra's Recursive Instantiation Mechanism
*****************************************

Hydra's instantiation mechanism is very powerful;
it is capable of recursively instantiating targets from nested configs.
hydra-zen's `builds` makes short work of generating nested configs.

.. code:: python

   # demonstrating recursive instantiation
   >>> Conf = builds(dict, x=builds(dict, x=builds(dict, x=[1, 2, 3])))
   >>> instantiate(Conf)
   {'x': {'x': {'x': [1, 2, 3]}}}

There are plenty of realistic examples where creating nested configurations is called for.
Consider this configuration of a data augmentation and transformation pipeline as an example:

.. code:: python

   from torchvision import transforms

   # imagine recreating this with seven manually-defined dataclasses...
   TrainTransformsConf = builds(
       transforms.Compose,
       transforms=[
           builds(transforms.RandomCrop, size=32, padding=4),
           builds(transforms.RandomHorizontalFlip, populate_full_signature=True),
           builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
           builds(transforms.RandomRotation, degrees=2),
           builds(transforms.ToTensor),
           builds(
               transforms.Normalize,
               mean=[0.4914, 0.4822, 0.4465],
               std=[0.2023, 0.1994, 0.2010],
           ),
       ],
   )

.. code:: python

   >>> instantiate(TrainTransformsConf)
   Compose(
       RandomCrop(size=(32, 32), padding=4)
       RandomHorizontalFlip(p=0.5)
       ColorJitter(brightness=[0.75, 1.25], contrast=[0.75, 1.25], saturation=[0.75, 1.25], hue=None)
       RandomRotation(degrees=[-2.0, 2.0], interpolation=nearest, expand=False, fill=0)
       ToTensor()
       Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
   )


Keeping DRY with Dynamically-Generated Configs 🌞
=================================================

**DRY – Don't Repeat Yourself** – is a `principle of software development <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`_ that cautions against repetitive software patterns and workflows.
Manually writing (or even statically generating) configs for your code will often leave you soaking **WET (Writing Everything Twice)**.

To see this, let's revisit the previous example where we configured the ``DNN`` class.

.. code:: python

   # Class that we want to configure
   class DNN:
       def __init__(
           self,
           input_size: int,
           output_size: int,
           layer_widths: Tuple[int, ...] = (5, 10, 5),
           device: str = "cpu",
       ):
           ...


Statically-Defined Configs are WET
**********************************

Manually writing a structured config for ``DNN`` entails hard-coding its current import-path as a string, and mirroring its signature:

.. code:: python

   # statically defined configurations are WET
   from omegaconf import MISSING
   from dataclasses import dataclass

   @dataclass
   class Builds_DNN:
      _target_: str = "vision.model.DNN"
      input_size: int = MISSING
      output_size: int = MISSING
      layer_widths: Tuple[int, ...] = (5, 10, 5),
      device: str = "cpu"

Doing this once isn't so cumbersome, but consider that any time we modify ``DNN`` by:

  - changing its location (e.g. move it to ``vision.classifiers.DNN``)
  - updating any of its parameters' names, default values, or annotations
  - adding or removing a parameter to/from its signature

then we will need to mirror this change in our configuration as well.
I hope you brought your towel, because things are getting WET 🌊.

Having to manually sync our configs with our code is not only tedious but it also creates a hot-spot for mistakes and bugs.


Generating Configs with `builds` Keeps Us DRY
*********************************************

We can stay nice and DRY by dynamically generating our configurations with `builds`


.. code:: python

   # dynamically-generated configurations are DRY
   from hydra_zen import builds
   from vision.model import DNN


   Builds_DNN = builds(DNN, populate_full_signature=True)

Here we don't need to worry about repeating ourselves by keeping our configuration in sync with our code: the structured config (complete with type annotations and default values) is *automatically* and *dynamically* generated for us at runtime! 🌞


What About Automatic Code Generation?
*************************************

Hydra provides a tool called `configen <https://github.com/facebookresearch/hydra/tree/master/tools/configen>`_ for automatically writing Python scripts with structured configs associated with your library's code.

.. code:: shell

   $ configen --config-dir conf --config-name configen
   my_lib.my_module -> /home/AlwaysWet/tmp/configen/example/config/configen/samples/my_module.py

This still means that we have to re-run this static code generation whenever we need to re-sync our configs with our updated code 🏊.

Generating static configs also has issues at-scale.
For example, `hydra-torch <https://github.com/pytorch/hydra-torch>`_ is a repository of statically generated configs for some parts of PyTorch's API.
While this is convenient to an extent, this repository of configs has to be:
 - actively maintained
 - versioned in-sync with PyTorch
 - included as an additional dependency in our projects

Furthermore, such repositories don't exist for most other libraries!
Thus this approach to code configuration is still a source of technical debt and repetitious work-flows.

With hydra-zen, we simply **configure what we need from any library** in an ergonomic, automatic, and dynamic way ⛱️.
