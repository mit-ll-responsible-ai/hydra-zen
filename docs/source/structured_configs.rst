#########################################
Dynamically Generating Structured Configs
#########################################

*************
API Reference
*************

hydra-zen provides us with some simple but powerful tools for dynamically generating structured configs for our code.
This helps to keep the process of configuring complex applications simple, intuitive, and free of technical debt.


.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   builds
   just
   hydrated_dataclass
   get_target


****************************
Basics of Structured Configs
****************************

Hydra supports configurations that are written in a yaml format or in Python via `structured configs <https://hydra.cc/docs/next/tutorials/structured_config/intro>`_.
Structured configs are `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ whose attributes store the configuration values, and whose type annotations (up to a limited assortment) can be leveraged by Hydra to provide runtime type checking of user-specified configuration values.

An important feature of these structured configs is that they too can be used in replacement of, and can be serialized to, yaml files by Hydra.
This is critical, as it ensures that each job that is launched using Hydra is fully documented by – and can be reproduced from – a plain-text yaml configuration.

A `targeted configuration <https://hydra.cc/docs/next/advanced/instantiate_objects/overview>`_ is designed to instantiate / call an object (a class-object or a function) with particular values.
hydra-zen provides functions that are specifically designed to dynamically generate targeted structured configs.
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
========================================

The following table compares the two standard approaches for configuring ``DNN`` – using a manually-written yaml file or a manually-defined structured config – along with hydra-zen's approach of **dynamically generating** a structured config.

+-------------------------------+---------------------------------------------------+------------------------------------------------------+
| (Hydra) Using a yaml file     | (Hydra) Using a structured config                 | (hydra-zen) Using `builds`                           |
+===============================+===================================================+======================================================+
| .. code:: yaml                | .. code:: python                                  | .. code:: pycon                                      |
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

Note that the result of ``builds(DNN, populate_full_signature=True)`` is equivalent to the manually-defined dataclass ``Builds_DNN``:
`builds` returns a dataclass object with parameters – along with their default values and type annotations – that are auto-populated based on the signature of ``DNN``.

.. code:: python

   >>> from hydra_zen import builds, instantiate
   >>> Builds_DNN = builds(DNN, populate_full_signature=True)

   >>> import inspect
   >>> inspect.signature(Builds_DNN)  # annotations and default values have been auto-populated
   <Signature (input_size: int, output_size: int, layer_widths: Tuple[int, ...] = (5, 10, 5), device: str = 'cpu') -> None>

   >>> conf_instance = Builds_DNN(input_size=2, output_size=10, device="cuda:0")
   >>> instantiate(conf_instance)  # "builds" DNN with particular config values
   DNN(input_size=2, output_size=10, layer_widths=[5, 10, 5], device=cuda:0)

And this dynamically generated configuration can still be serialized to a yaml file by Hydra:

.. code:: pycon

   >>> from hydra_zen import to_yaml  # alias of `omegaconf.OmegaConf.to_yaml`
   >>> print(to_yaml(conf_instance))
   _target_: vision.model.DNN
   input_size: 2
   output_size: 10
   layer_widths:
   - 5
   - 10
   - 5
   device: cuda:0


.. _DRY:

*************************************************
Keeping DRY with Dynamically-Generated Configs 🌞
*************************************************

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
==================================

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
=============================================

We can stay nice and DRY by dynamically generating our configurations with `builds`


.. code:: python

   # dynamically-generated configurations are DRY
   from hydra_zen import builds
   from vision.model import DNN


   Builds_DNN = builds(DNN, populate_full_signature=True)

Here we don't need to worry about repeating ourselves by keeping our configuration in sync with our code: the structured config (complete with type annotations and default values) is *automatically* and *dynamically* generated for us at runtime! 🌞


What About Automatic Code Generation?
=====================================

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
With hydra-zen, **we simply configure what we need from any library** in an ergonomic, automatic, and dynamic way ⛱️.

.. _Builds:

**************************
The Essentials of `builds`
**************************

Learning the essentials of `builds` is all the average user needs to use it in their project.
You likely have already gleaned most of its essential functionality from the examples that you have read thus far,
but it is worthwhile for us to discuss these more deliberately.

`builds` generates dataclass objects
====================================

`builds` dynamically generates a dataclass object that configures a particular target.
Let's generate a targeted configuration for the following function

.. code:: python

   # contents of my_funcs.py

   from typing import List

   def make_a_dict(x: int, y: List[int]):
       return {"x": x, "y": y}


Here is what `builds` effectively defines for us "under the hood"

+---------------------------------------------------+--------------------------------------------------------------------------+
| Example Using `builds`                            | Equivalent dataclass                                                     |
+===================================================+==========================================================================+
| .. code:: pycon                                   | .. code:: python                                                         |
|                                                   |                                                                          |
|    >>> from hydra_zen import builds               |    from dataclasses import dataclass, field                              |
|    >>> builds(make_a_dict, x=2, y=[1, 2, 3])      |                                                                          |
|    types.Builds_make_a_dict                       |    @dataclass                                                            |
|                                                   |    class Builds_dict:                                                    |
|                                                   |        _target_: str = field(default='my_funcs.make_a_dict', init=False) |
|                                                   |        x: int = 2                                                        |
|                                                   |        y: List[int] = field(default_factory=lambda: list([1, 2, 3]))     |
|                                                   |                                                                          |
+---------------------------------------------------+--------------------------------------------------------------------------+

Note how `builds` handles for us the cumbersome tasks of `safely setting mutable default values <https://docs.python.org/3/library/dataclasses.html#mutable-default-values>`_, excluding Hydra-specific "hidden" parameters from the class-init signature,
and mirroring the target's type annotations.
As we study more of `builds`'s features, we will see that there are many "wins" that we will enjoy by leveraging this function to generate our configurations.

The resulting `dataclass object <https://docs.python.org/3/library/dataclasses.html>`_ is designed specifically to behave as `targeted structured configs  <https://hydra.cc/docs/next/advanced/instantiate_objects/overview>`_ that can be instantiated by Hydra.

.. code:: pycon

   >>> from dataclasses import is_dataclass
   >>> Builds_dict = builds(make_a_dict, x=2, y=[1, 2, 3])  # signature: Builds_dict(x: Any = 2, y: Any = <factory>)
   >>> is_dataclass(Builds_dict)
   True

   # creating an instance of the dataclass with an updated configuration
   >>> conf_instance = Builds_dict(x=-100)
   >>> conf_instance
   Builds_make_a_dict(_target_='my_funcs.make_a_dict', x=-100, y=[1, 2, 3])

   >>> conf_instance.x
   -100
   >>> conf_instance.y
   [1, 2, 3]

   >>> from hydra_zen import instantiate  # annotated alias of `hydra.utils.instantiate`
   >>> instantiate(conf_instance)  # calls `dict(x=-100, y=[1, 2, 3])`
   {'x': -100, 'y': [1, 2, 3]}

Accordingly, `builds` accepts Hydra-specific parameters for tuning the behavior of the structured config (e.g. disabling `recursive instantiation <https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation>`_).


.. _Partial:

Configuring a Target for Partial Instantiation
==============================================

`builds` is capable for configuring a target such that it will only be *partially* instantiated.
This is very useful, as it is often the case that our configuration of a target can only partially describe its input parameters.

For example, ``Adam`` is an gradient-based optimizer that is popular in the PyTorch library and that will frequently appear as a configurable component to a deep learning experiment.
This optimizer has configurable parameters, such as a "learning rate" (``lr``), which must be provided by the user in order to instantiate ``Adam``.
The optimizer must *also* be initialized with the parameters that it will optimizing, however these parameters are typically created after we have started running our program and thus *they are not available to / cannot be part of our configuration*.
Fortunately, we can use ``builds(..., zen_partial=True)`` to configure ``Adam`` to be only partially-built using those values that we have access to at configuration time.

.. code:: pycon

   >>> from torch.optim import Adam
   >>> from torch import tensor

   >>> PartialBuilds_Adam = builds(Adam, zen_partial=True, lr=10.0)
   >>> partial_optim = instantiate(PartialBuilds_Adam)
   >>> partial_optim  # a partially-instantiated Adam
   functools.partial(<class 'torch.optim.adam.Adam'>, lr=10.0)


As promised, instantiating this config only partially-builds the ``Adam`` optimizer; we can finish instantiating it once we have access to our model's parameters

.. code:: pycon

   >>> model_params = [tensor(1.0), tensor(-2.0)]
   >>> partial_optim(model_params)
   Adam (
   Parameter Group 0
       amsgrad: False
       betas: (0.9, 0.999)
       eps: 1e-08
       lr: 10.0
       weight_decay: 0
   )

.. note::
   Leveraging ``builds(..., zen_partial=True)`` will produce a config that depends explicitly
   on hydra-zen. I.e. hydra-zen must be installed in order to instantiate the resulting config.

.. _Auto:

Combining Auto-Populated and User-Specified Default Values
==========================================================


We can feed `builds` values for a subset of the target's parameters and then auto-populate the remaining parameters.

.. code:: python

   def func(x: int = 2, y: str = "hi"):
       ...

   # signature: `Builds_func(x: int = 10, y: str = "hi")`
   builds(func, x=10, populate_full_signature=True)

Or, we can intentionally exclude a target's parameter so that it is not available for configuration

.. code:: python

   # signature: `Builds_func(y: str = "dinosaur")`
   builds(func, y="dinosaur")  # `x` is not configurable for building `func`


We can provide positional arguments as well

.. code:: pycon

   >>> instantiate(builds(np.add, 1.0, 2.0))
   3.0


Arguments specified in a positional way are excluded from the dataclass' signature

.. code:: python

   # signature: `Builds_func(y: str = 'a string')`
   builds(func, 2, y="a string")



Nesting configs for recursive instantiation
===========================================

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


Auto-generated nested configs
-----------------------------

`builds` will automatically generate nested configurations in order to configure a target and its signature's default values.
E.g. if a default-value in a target's signature is a function, then `builds` will create a dataclass that configures
this function and then nest it within the target's configuration.

.. code:: python

   # The default value for `reduction_fn` needs its own structured config
   # in order for it to be included in this function's configuration
   def objective_function(prediction, score, reduction_fn=np.mean):
        ...

.. code:: pycon

   >>> Builds_loss = builds(objective_function, populate_full_signature=True)

   >>> Builds_loss.reduction_fn  # `just(np.mean)` was automatically created
   types.Just_mean

   >>> print(to_yaml(Builds_loss))
   _target_: __main__.objective_function
   prediction: ???
   score: ???
   reduction_fn:
     _target_: hydra_zen.funcs.get_obj
     path: numpy.mean


User-specified parameters will automatically be transformed as well

.. code:: pycon

   >>> print(to_yaml(builds(dict, io_fn=len, sequence=list)))
   _target_: builtins.dict
   io_fn:
     _target_: hydra_zen.funcs.get_obj
     path: builtins.len
   sequence:
     _target_: hydra_zen.funcs.get_obj
     path: builtins.list

.. note::
   Leveraging this feature will produce a config that depends explicitly
   on hydra-zen. I.e. hydra-zen must be installed in order to instantiate the resulting config.


Creating immutable configs
==========================

Dataclasses can be made to be "frozen" such that their instances are immutable.
Thus we can use `builds` to create immutable configs.

.. code:: pycon

   >>> RouterConfig = builds(dict, ip_address=None, frozen=True)
   >>> my_router = RouterConfig(ip_address="192.168.56.1")  # an immutable instance
   >>> my_router.ip_address = "a bad address"
   ---------------------------------------------------------
   FrozenInstanceError: cannot assign to field 'ip_address'


Composing configs via inheritance
=================================

The ``builds_bases`` argument enables us to compose configurations using inheritance

.. code:: pycon

   >>> ParentConf = builds(dict, a=1, b=2)
   >>> ChildConf = builds(dict, b=-2, c=-3, builds_bases=(ParentConf,))

   >>> instantiate(ChildConf)
   {'a': 1, 'b': -2, 'c': -3}

   >>> issubclass(ChildConf, ParentConf)
   True


**********************************
The Bells and Whistles of `builds`
**********************************

There is more to `builds` than meets the eye; the following "bells and whistles" of `builds` are nice to know, but are not essential to using hydra-zen.


Runtime validation
==================

Misspelled parameter names and other invalid configurations for the target's signature will be caught by `builds`, so that the error surfaces immediately while creating the configuration.

.. code:: pycon

   >>> def func(a_number: int): pass
   >>> builds(func, a_nmbr=2)  # misspelled parameter name
   ---------------------------------------------------------------------------
   TypeError: Building: func ..
   The following unexpected keyword argument(s) was specified for __main__.func via `builds`: a_nmbr

   >>> BaseConf = builds(func, a_number=2)
   >>> builds(func, 1, builds_bases=(BaseConf,))  # bad composition is caught upon construction
   ---------------------------------------------------------------------------
   TypeError: Building: func ..
   multiple values for argument a_number were specified for __main__.func via `builds`

   >>> builds(func, 1, 2)  # too many arguments
   ---------------------------------------------------------------------------
   TypeError: Building: func ..
   __main__.func takes 1 positional args, but 2 were specified via `builds`


Because `builds` automatically mirrors type annotations from the target's signature, we also benefit from Hydra's type-validation mechanism.


.. code:: pycon

   >>> def func(parameter: int): pass
   >>> instantiate(builds(func, parameter="a string"))
   ---------------------------------------------------------------------------------
   ValidationError: Invalid value assigned : str is not a int.

Automatic Type Refinement
=========================

Hydra permits only `a narrow subset of type annotations <https://hydra.cc/docs/next/tutorials/structured_config/intro#structured-configs-supports>`_ to be present in a target's signature:

   - ``Any``
   - Primitive types (``int``, ``bool``, ``float``, ``str``, ``Enums``)
   - Nesting of Structured Configs
   - Containers (List and Dict) containing primitives or Structured Configs
   - Optional fields

Annotations that fall outside of this subset will cause Hydra's runtime validation to raise an error.
As such, `builds` will automatically "broaden" the annotations associated with a target's signature so that it will be made compatible with Hydra.
For example, suppose that we want to configure

.. code:: python

   # `Literal[...]` is not supported by Hydra
   def func(ones_and_twos: List[Literal[1, 2]]):
       ...

``Literal[1, 2]`` is not supported by Hydra, so `builds` will "broaden" this type to ``Any``.

.. code:: python

   # signature: `Builds_func(ones_and_twos: List[Any] = <factory>)`
   Builds_func = builds(func, ones_and_twos=[1, 2])

In this way, we can still configure and build this function, but we also retain some level of type-validation

.. code:: pycon

   >>> instantiate(Builds_func("not a list"))
   ---------------------------------------------------------------------------------
   ValidationError: Invalid value assigned : str is not a ListConfig, list or tuple.
    full_key:
    object_type=None

In general, hydra-zen will broaden types as-needed so that dynamically-generated configs will never include annotations that would cause Hydra to error-out.


Combining Static and Dynamic Configurations with `@hydrated_dataclass`
======================================================================

hydra-zen provides a decorator, `hydrated_dataclass`, which is similar to the standard `@dataclass` but can be used to auto-populate Hydra-specific parameters;
it also exposes other features that are available in `builds`.

.. code:: python

   from hydra_zen import hydrated_dataclass

   from torch.optim import Adam

   @hydrated_dataclass(target=Adam, zen_partial=True, frozen=True)
   class BuildsAdam:
       lr: float = 0.01
       momentum: float = 0.9

   BuildsAdam(lr="a string")  # static type-checker flags as invalid (invalid type)

   conf = BuildsAdam()
   conf.lr = 10.0  # static type-checker flags as invalid (mutating "frozen" dataclass)


This has the benefit of making certain pertinent information (e.g. the dataclass' fields and that it is frozen) available to static type checkers, while still dynamically populating the resulting dataclass with Hydra-specific fields (e.g. ``_target_`` and ``_partial_target_``) and providing the same runtime validation as `builds`.

Note that the ``@hydrated_dataclass`` decorator uses a `recently proposed <https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md>`_ mechanism for enabling static tools to "recognize" third-party dataclass decorators like this one.
Presently, the above static inspection is only supported by pyright, but other type-checkers will likely add support for this soon.


hydra_zen.typing
================

``hydra_zen.typing`` ships generic protocols that annotate the outputs of `builds` and `just`.
These annotations are used to overload ``hydra_zen.instantiate`` so that static type checkers can "see" what is being instantiated.
The following code block uses comments to indicate the types that will be inferred by static type checkers.

.. code:: python

   class MyClass:
       def __init__(self, x: int):
           pass

   Conf = builds(MyClass, x=1)  # type: Type[Builds[Type[MyClass]]]
   conf = Conf(x=10)            # type: Builds[Type[MyClass]]

   my_class1 = instantiate(Conf)  # type: MyClass
   my_class2 = instantiate(conf)  # type: MyClass

   PartialConf = builds(MyClass, zen_partial=True)  # type: Type[PartialBuilds[Type[MyClass]]]
   partial_conf = PartialConf()                       # type: PartialBuilds[Type[MyClass]]

   partiald_class = instantiate(PartialConf)   # type: Partial[MyClass]
   my_class3 = partiald_class(x=2)             # type: MyClass

   partiald_class2 = instantiate(partial_conf) # type: Partial[MyClass]
   my_class4 = partiald_class2(x=2)            # type: MyClass

   JustMyClass = just(MyClass)               # type: Type[Just[Type[MyClass]]]
   my_class_type = instantiate(JustMyClass)  # type: Type[MyClass]

(Note that this behavior is verified using the static type checker `pyright <https://github.com/microsoft/pyright>`_, which is used by VSCode.
PyCharm's type-checker appears to struggle with deeply-nested types like ``Type[Builds[Type[MyClass]]]``, but this is an issue on their end.)