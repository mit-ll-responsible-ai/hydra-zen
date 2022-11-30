.. hydra-zen documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. admonition:: Join the Discussion 💬

   Share ideas, ask questions, and chat with us over at `hydra-zen's discussion board <https://github.com/mit-ll-responsible-ai/hydra-zen/discussions>`_.


.. tip::

   🎓 Using hydra-zen for your research project? `Cite us <https://arxiv.org/abs/2201.05647>`_!


=====================================
Welcome to hydra-zen's documentation!
=====================================

hydra-zen is a Python library that makes the `Hydra framework <https://hydra.cc/>`_  
simpler and more elegant to use. Use hydra-zen to design your project to be:

- **Configurable**: change deeply-nested parameters and swap out entire pieces of your program, all from the command line. 
- **Repeatable**: each run of your code will be self-documenting; the full configuration of your software is saved alongside your results.
- **Scalable**: launch multiple runs of your software, be it on your local machine or across multiple nodes on a cluster.

It provides functions that :ref:`dynamically <create-config>` and :ref:`automatically <additional-types>` generate configs for your code, which help to eliminate 
Hydra-specific boilerplate from your project. 

hydra-zen is fully compatible with Hydra, and is appropriate for use in both rapid 
prototypes and production-grade code. It is also great for designing your data science 
and machine learning research to be reproducible. hydra-zen provides specialized 
support for using NumPy, Jax, PyTorch, and Lightning (a.k.a PyTorch-Lightning) in your 
Hydra application. 


.. admonition:: Attention, Hydra users:

  If you are already use Hydra, let's cut to the chase: **the most important benefit of using hydra-zen is that it automatically and dynamically generates structured configs for you**.
  

  .. code-block:: python
     :caption: Creating a structured config *without hydra-zen*
     
     from dataclasses import dataclass, field
     
     def foo(bar: int, baz: list[str], qux: float = 1.23):
         ...
     
     @dataclass
     class FooConf:
         _target_: str = "__main__.foo"
         bar: int = 2
         baz: list[str] = field(default_factory=lambda: ["abc"])
         qux: float = 1.23


  .. code-block:: python
     :caption: Creating an equivalent structured config *with hydra-zen*

     from hydra_zen import builds

     def foo(bar: int, baz: list[str], qux: float = 1.23):
         ...

     ZenFooConf = builds(foo, bar=2, baz=["abc"], populate_full_signature=True)
  
  You can even write dataclasses that aren't natively compatible with Hydra and then use hydra-zen to dynamically convert them to Hydra-compatible structured configs.

  .. code-block:: python
     :caption: Converting a general dataclass to a Hydra-compatible structured config
  
     from dataclasses import dataclass
     from typing import Callable, Sequence
     
     from hydra_zen import just, instantiate, to_yaml

     @dataclass
     class Bar:
        # annotation not supported by Hydra
        reduce_fn: Callable[[Sequence[float]], float]
  
     bar = Bar(reduce_fn=sum)  # value not supported by Hydra
      
     compat_bar = just(bar)  # returns a Hydra-compatible structured config!
      
     assert to_yaml(compat_bar) == """\
     _target_: __main__.Bar
     reduce_fn:
       _target_: hydra_zen.funcs.get_obj
       path: builtins.sum
     """

     inst_bar = instantiate(compat_bar)
     assert isinstance(inst_bar, Bar)
     assert inst_bar == bar

  This means that it is much **easier and safer** to write and maintain the configs for your Hydra applications:
  
  - Write all of your configs in Python. No more yaml files!
  - Write less, :ref:`stop repeating yourself <dry>`, and get more out of your configs.
  - Get automatic type-safety via :func:`~hydra_zen.builds`'s signature inspection.
  - :ref:`Validate your configs <builds-validation>` before launching your application.
  - Leverage :ref:`auto-config support <additional-types>` for additional types, like :py:class:`functools.partial`, that are not natively supported by Hydra.

hydra-zen also also provides Hydra users with powerful, novel functionality. With it, we can:

- Add :ref:`enhanced runtime type-checking <runtime-type-checking>` for our Hydra application, via :ref:`pydantic <pydantic-support>`, `beartype <https://github.com/beartype/beartype>`_, and other third-party libraries.
- Design configs specialized behaviors, like :ref:`configs with meta-fields <meta-field>`. 
- Leverage a powerful :ref:`functionality-injection framework <zen-wrapper>` in our Hydra applications.
- Run static type-checkers on our config-creation code to catch incompatibilities with Hydra.


Installation
============

hydra-zen is lightweight: its only dependencies are ``hydra-core`` and 
``typing-extensions``. To install it, run:

.. code:: console

   $ pip install hydra-zen

If instead you want to try out the features in the upcoming version, you can install 
the latest pre-release of hydra-zen with:

.. code:: console

   $ pip install --pre hydra-zen


Learning About hydra-zen
========================

Our docs are divided into four sections: Tutorials, How-Tos, Explanations, and 
Reference.

If you want to get a bird's-eye view of what hydra-zen is all about, or if you are 
completely new to Hydra, check out our **Tutorials**. For folks who are savvy Hydra 
users, our **How-Tos** and **Reference** materials can help acquaint you with the 
unique capabilities that are offered by hydra-zen. Finally, **Explanations** provide 
readers with taxonomies, design principles, recommendations, and other articles that 
will enrich their understanding of hydra-zen and Hydra.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   how_tos
   explanation
   api_reference
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
