***************************
Creating Structured Configs
***************************

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

A `targeted configuration <https://hydra.cc/docs/next/advanced/instantiate_objects/overview>`_ is designed to instantiate / call an object (a class-object or a function) with particular values.
hydra-zen provides functions that are specifically designed to created targeted structured configs.

For example, suppose that we want to configure the following class

.. code:: python

   # contents of vision/model.py

   from typing import Tuple

   class DNN:
       def __init__(
          self,
          input_size: int,
          output_size: int,
          layer_widths: Tuple[int, ...] = (5, 10, 5),
          device: str = "cpu",
       ):
           print(input_size, output_size, layer_widths, device)


+----------------------------+----------------------------------------+-------------------------------------------------+
| Using a yaml file          | Using a structured config              | Using hydra-zen                                 |
+============================+========================================+=================================================+
| .. code:: yaml             | .. code:: python                       | .. code:: python                                |
|                            |                                        |                                                 |
|    _target_: mylib.my_func |    @dataclass                          |    # creates equivalent dataclass               |
|    number: 1               |    class Builds_my_func:               |    >>> builds(my_func, a_number=1, text="hello")|
|    text: hello             |        _target_ : str = "mylib.my_func"|    types.Builds_my_func                         |
|                            |        a_number : int = 1              |                                                 |
|                            |        text : str = "hello"            |                                                 |
|                            |                                        |                                                 |
+----------------------------+----------------------------------------+-------------------------------------------------+





