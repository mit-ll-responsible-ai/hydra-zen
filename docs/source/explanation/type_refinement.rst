.. _type-support:

############################################################
Understanding hydra-zen's Automatic Type Refinement Behavior
############################################################

hydra-zen's :ref:`config-creation functions <create-config>` will automatically 
refine user-supplied type annotations so that they are made compatible with Hydra's 
limited type support. This eliminates a friction that will otherwise force users to 
modify their code's type annotations. Furthermore, hydra-zen preserves 
type information such that third-party type checkers can be used to provide more 
robust support for more general type annotations.

Hydra's Limited Type Support
----------------------------

Hydra permits only `a narrow subset of type annotations <https://hydra.cc/docs/tutorials/structured_config/intro#structured-configs-supports>`_ to be present in a 
config:

   - ``Any``
   - Primitive types (``int``, ``bool``, ``float``, ``str``, ``Enums``)
   - Nesting of Structured Configs
   - Containers (List and Dict) containing primitives or Structured Configs
   - Optional fields
   - Nested containers (*added in OmegaConf v2.2.0*)

Annotations that fall outside of this subset will cause Hydra to raise an error. 
This means that the following config:

.. code-block:: python

   from typing_extensions import Literal
   from dataclasses import dataclass
   
   @dataclass
   class A:
       x: Literal[1, 2]

will cause Hydra to raise an error during instantiation, due to the presence of an
unsupported type-annotation (``Literal[1, 2]``). E.g.

.. code-block:: pycon

    >>> from hydra_zen import instantiate
    >>> instantiate(A, x=1)
    ConfigTypeError   Traceback (most recent call last) [...]

This behavior can be highly undesirable as it requires users to modify their code and 
use less-accurate type-annotations.

Leveraging Automatic Type-Refinement (without having to think about it)
-----------------------------------------------------------------------

To eliminate the friction that occurs here, hydra-zen's :ref:`config-creation functions <create-config>` will automatically-broaden the type annotations on the configs that they produce. This means that users will not need to change their 
type-annotations for the sake of Hydra. 

For example, to address the incompatibility seen in the previous example, we can use 
:func:`~hydra_zen.builds`.

.. code-block:: pycon

    >>> from hydra_zen import builds
    >>> BuildsA = builds(A, populate_full_signature=True)
    >>> instantiate(BuildsA, x=1)
    A(x=1)


This type-broadening behavior will try to preserve as much type information as it can
without creating compatibility issues with Hydra. For example, suppose that we want to 
configure the following function:

.. code-block:: python

   from typing import List

   def func(ones_and_twos: List[Literal[1, 2]]):
       return ones_and_twos

As we saw above, ``Literal[1, 2]`` is not supported by Hydra. That being said, 
``List`` *is* supported, thus :func:`~hydra_zen.builds` will "broaden" 
``List[Literal[1, 2]]`` to ``List[Any]``.

.. code-block:: python

   # signature: `Builds_func(ones_and_twos: List[Any])`
   Builds_func = builds(func, populate_full_signature=True)

In this way, we can still configure and build this function, but we also retain some level of type-validation

.. code-block:: pycon

   >>> instantiate(Builds_func, ones_and_twos="not a list")
   ---------------------------------------------------------------------------------
   ValidationError: Invalid value assigned : str is not a ListConfig, list or tuple.
    full_key:
    object_type=None

   >>> instantiate(Builds_func, ones_and_twos=[1, 2, 3])
   [1, 2, 3]
   
In general, hydra-zen will broaden types as-needed so that dynamically-generated configs will never include annotations that would cause Hydra to raise an error due
to lack of support for that type.

.. _pydantic-support:

Using Third-Party Runtime Type-Checkers
---------------------------------------
Although hydra-zen will broaden the types that get exposed to Hydra, the original 
type-information of a target that is provided to :func:`~hydra_zen.builds` is still
preserved. This means that third-party type checkers like 
`pydantic <https://pydantic-docs.helpmanual.io/>`_ and 
`beartype <https://github.com/beartype/beartype>`_ can be used to provide higher quality
type-checking functionality.

E.g. let's return to the original example involving the dataclass ``A``. Assuming that
we have installed ``pydantic``, we can use it to recreate this dataclass so that it 
will perform general, runtime type-checking for us.

.. code-block:: python

   from pydantic.dataclasses import dataclass as pyd_dataclass
    
   @pyd_dataclass
   class A:
       x: Literal[1, 2]

   BuildsA = builds(A, populate_full_signature=True)

As we saw earlier, Hydra will no longer complain about this type-annotation.

.. code-block:: pycon

    >>> instantiate(BuildsA, x=1)
    A(x=1)

But now ``pydantic`` will actually ensure that ``x`` is either ``1`` or ``2``.

.. code-block:: pycon

    >>> instantiate(BuildsA, x=-10)
    ValidationError: 1 validation error for A
    x
    unexpected value; permitted: 1, 2 (type=value_error.const; given=-10; permitted=(1, 2)  )

hydra-zen also provides support for leveraging these third-party type-checkers 
directly, via the ``zen_wrappers`` feature of :func:`~hydra_zen.builds`. See 
:ref:`data-val` for more details.
