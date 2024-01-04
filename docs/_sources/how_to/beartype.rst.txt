.. meta::
   :description: hydra-zen provides a wrapped-instantiation mechanism that enables the use runtime type-checkers, like pydantic and beartype, with Hydra applications.

.. admonition:: Prerequisites

   Your must install `beartype <https://github.com/beartype/beartype>`_ in your Python environment in order to complete this How-To guide.

.. _runtime-type-checking:

=================================================
Add Enhanced Runtime Type-Checking to a Hydra App
=================================================

Many Python libraries (e.g. NumPy, PyTorch) include so-called :plymi:`type hints <Module5_OddsAndEnds/Writing_Good_Code.html#Type-Hinting>` in the signatures of their 
functions and classes; a type-hint documents the type of values that an input 
parameter expects. Let's turn these hints into promises! We can modify the configs in 
our Hydra app so that its configured values are verified against these types at runtime.

In this How-To guide we will:

1. Install a lightweight runtime type-checker, `beartype <https://github.com/beartype/beartype>`_.
2. Create "toy" library code that includes type annotations.
3. Design configs for a Hydra app that will leverage beartype's type-checking throughout its interface.
4. Test that bad input values get "caught" by our type-checker.

.. tip::
   
   hydra-zen also provides support for the popular runtime type-checker `pydantic <https://pydantic-docs.helpmanual.io/>`_. It can be used instead of beartype. 
   Consult the reference for :func:`~hydra_zen.third_party.pydantic.validates_with_pydantic` to adapt this How-To accordingly.

To begin, we will install beartype in our Python environment.

.. code-block:: console
   :caption: 1: Installing beartype.

   $ pip install beartype

Next, let's create a toy example of some library code that uses type hints, but does 
not leverage runtime checking of it types. Create a script called ``toy_library.py``, 
containing the following code.

.. code-block:: python
   :caption: 2: Creating toy example of library code. Contents of ``toy_library.py``.

   from typing import Union, Sequence, TypeAlias
   
   from beartype.vale import Is
   from typing_extensions import Annotated
   
   
   PositiveInt: TypeAlias = Annotated[int, Is[lambda x: x >= 0]]
   
   
   def process_age(age: PositiveInt):
       return age
   
   
   def process_shape(shape: Union[int, Sequence[int]]):
       return shape

Note that the annotation `PositiveInt` indicates that an associated value should not only be an :class:`int`, but also have a non-negative value. Whereas

.. code-block:: python
   :caption: Annotation of ``shape``

   Union[int, Sequence[int]]

indicates that a value should either be an :class:`int` or any sequence (list, tuple, 
etc.) of ints.

Supposing that our Hydra app will configure these two "toy" functions, let's design 
their configs so that beartype will validate their configured values upon 
instantiation.

Open a Python console -- or Jupyter notebook -- in the same directory as ``toy_library.py`` and create the following configs.

.. code-block:: pycon
   :caption: 3: Creating configs that include beartype validation.

   >>> from toy_library import process_age, process_shape

   >>> from hydra_zen import make_custom_builds_fn
   >>> from hydra_zen.third_party.beartype import validates_with_beartype

   >>> builds = make_custom_builds_fn(
   ...     populate_full_signature=True,
   ...     zen_wrappers=validates_with_beartype,
   ...     hydra_convert="all",
   ... )
   
   >>> ConfAge = builds(process_age)
   >>> ConfShape = builds(process_shape)

Finally, let's check that our configured values are validated as-expected.
In the same console, verify that you can replicate the following behavior.

.. code-block:: pycon
   :caption: 4: Test that the type-checker catches bad configured values.

   >>> from hydra_zen import instantiate

   >>> instantiate(ConfAge, age=12)  # OK
   12
   
   >>> instantiate(ConfAge, age=-100)  # Bad: negative int
   BeartypeCallHintPepParamException- process_age() parameter age=-100 violates type 
   hint [...]

   >>> instantiate(ConfAge, age="twelve")  # Bad: not an int
   BeartypeCallHintPepParamException- process_age() parameter age='twelve' violates 
   type hint [...]
   
   >>> instantiate(ConfShape, shape=3)  # OK
   3
   
   >>> instantiate(ConfShape, shape=[1, 2, 5])  # OK
   [1, 2, 5]
   
   >>> instantiate(ConfShape, shape=["a", "b"])  # Bad: not a sequence of ints
   BeartypeCallHintPepParamException- process_shape() parameter shape=['a', 'b'] 
   violates type hint [...]

Awesome! Now mis-configured values will have a bear of a time getting past our app's 
type-checked interface 🐻.

.. admonition:: References

   - :func:`~hydra_zen.third_party.beartype.validates_with_beartype`
   - :ref:`type-support`

