.. meta::
   :description: Adding pydantic parsing to a Hydra app

.. admonition:: tl;dr

   When wrapping your main function with `hydra_zen.zen`, pass in `hydra_zen.third_party.pydantic.pydantic_parser` to the `instantiation_wrapper` argument of 
   `~hydra_zen.zen`. Additionally, for any `hydra_zen.instantiate` calls you make, pass 
   this parser in as the `_target_wrapper_` argument.


.. admonition:: Prerequisites

   Your must install `pydantic <https://docs.pydantic.dev/latest/>`_ in your Python environment in order to complete this How-To guide.

.. _pydantic-parsing:

========================================================
Add Pydantic Type Checking and Parsing to Your Hydra App
========================================================

Hydra's runtime type checking is :ref:`limited to only a narrow subset of Python's typing features <hydra-type-support>`. By contrast, `Pydantic <https://pydantic-docs.helpmanual.io/>`_ provides much more comprehensive type checking 
capabilities; additionally, it is capable of parsing CLI inputs into complex data 
structures (e.g., convert a string to a :py:class:`pathlib.Path`). In this how-to 
guide, we will **add pydantic parsing to all config-instantiation sites in our 
Hydra-app**.


Consider the following simple hydra-zen app:


.. code-block:: python
   :caption: Contents of `my_app.py`

   from hydra_zen import store
   from hydra_zen.third_party.pydantic import pydantic_parser
   
   from typing import Literal
   from pydantic import PositiveInt
   
   
   from dataclasses import dataclass
   
   
   @dataclass
   class Character:
       age: PositiveInt = 22
       name: str = "Bobby"
   
   
   def main(
       character: Character,
       gear: tuple[str, ...] = (),
       mode: Literal["easy", "hard"] = "easy",
   ):
       print(f"{character=!r} {gear=!r}  {mode=!r}")
   
   
   if __name__ == "__main__":
       from hydra_zen import zen
   
       store(main, hydra_defaults=["_self_", {"character": "base"}])
       store(Character, group="character", name="base")
       store.add_to_hydra_store()
   
       zen(
           main,
           # This is the key ingredient
           instantiation_wrapper=pydantic_parser,
       ).hydra_main(
           config_name="main",
           config_path=None,
           version_base="1.3",
       )


By specifying `instantiation_wrapper=pydantic_parser` in our `hydra_zen.zen` call, we 
insure that all config-instantiation sites in our Hydra-app will use pydantic parsing. 
Because of this, we will benefit from the following features:

   1. Annotating `mode` with `Literal` will restrict the permitted values for this parameter.
   2. We can configure `gear` with a list of strings at the CLI and - because of the `tuple[str, ...]` annotation - it will be coerced into a tuple of strings before being passed to `main`.
   3. We can use a `pydantic.PositiveInt` annotation for the `character.age` nested field; this will ensure that the age is a valid value.


Let's check that `mode` is restricted to the values `easy` and `hard`:

.. code-block:: console
   :caption: Running the app with an invalid value for `mode`

   $ python my_app.py mode='medium'
   Traceback (most recent call last):
     ...
   pydantic_core._pydantic_core.ValidationError: 1 validation error for main 
   mode
   Input should be 'easy' or 'hard'

Next, let's see that `gear` can be passed a list, which will be coerced into a tuple:

.. code-block:: console
   :caption: Running the app with a list of strings for `gear`

   $ python my_app.py gear='[sword,shield]'
   character=Character(age=22, name='Bobby') gear=('sword', 'shield')  mode='easy'


Finally, let's see that `character.age` is correctly parsed as a `PositiveInt`:

.. code-block:: console
   :caption: Running the app with an invalid value for `character.age`

   $ python my_app.py character.age=-1
   Traceback (most recent call last):
     ...
   Error in call to target '__main__.Character':
   1 validation error for Character
   age
     Input should be greater than 0 

In this way, we can use Pydantic to add powerful type checking and parsing to our Hydra 
apps.

If your app includes manual calls to `hydra_zen.instantiate`, you can also pass in the
`pydantic_parser` as the `_target_wrapper_` argument to ensure that these 
config-instantiation calls also use Pydantic parsing.

.. note::

   Keep in mind that this pydantic parsing layer only occurs when we instantiate
   configs that have `_target_` fields, and that it uses the annotations of the
   `_target_` objects.