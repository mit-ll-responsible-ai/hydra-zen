
Creating Structured Configs
===========================

These configurations can be written in a yaml format, or in Python via `structured configs <https://hydra.cc/docs/next/tutorials/structured_config/intro>`_.
Structured configs are dataclasses whose type annotations (up to a limited assortment) can be leveraged by Hydra to provide runtime type checking for your configurations.


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



.. currentmodule:: hydra_zen

.. autosummary::
   :toctree: generated/

   builds
   just
   hydrated_dataclass


