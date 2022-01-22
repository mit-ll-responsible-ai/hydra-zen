.. hydra-zen documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
Welcome to hydra-zen's documentation!
=====================================

hydra-zen is a Python library that simplifies the process of writing code (be it research-grade or production-grade) that is:

- **Configurable**: you can configure all aspects of your code from a single interface (the command line or a Python function).
- **Repeatable**: each run of your code will be self-documenting; the full configuration of your software is saved alongside your results.
- **Scalable**: launch multiple runs of your software, be it on your local machine or across multiple nodes on a cluster.

It builds off -- and is fully compatible with -- `Hydra <https://hydra.cc/>`_, a 
framework for elegantly configuring complex applications. hydra-zen helps simplify the 
process of using Hydra by providing specialized functions for :ref:`creating configs <create-config>` and 
launching Hydra jobs. 

hydra-zen also also provides Hydra users with powerful, novel functionality. With it, we can:

- Add :ref:`enhanced runtime type-checking <runtime-type-checking>` for our Hydra application, via :ref:`pydantic <pydantic-support>`, beartype, and other third-party libraries.
- Design configs with specialized behaviors, like :ref:`partial configs <partial-config>`, or :ref:`configs with meta-fields <meta-field>`. 
- Use :ref:`additional data types <additional-types>` in our configs, like :py:class:`pathlib.Path`, that are not natively supported by Hydra.
- :ref:`Validate our configs at runtime <builds-validation>`, prior to launching our application.
- Leverage a powerful :ref:`functionality-injection framework <zen-wrapper>` in our Hydra applications.
- Run static type-checkers on our config-creation code to catch incompatibilities with Hydra.



.. admonition:: Join the Discussion 💬

   Share ideas, ask questions, and chat with us over at `hydra-zen's discussion board <https://github.com/mit-ll-responsible-ai/hydra-zen/discussions>`_.


.. tip::

   🎓 Using hydra-zen for your research project? `Cite us <https://zenodo.org/record/5584711>`_!

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


hydra-zen at a Glance
=====================

Suppose we want to use Hydra to create a Python application that can be configured and launched from the command line.
hydra-zen provides elegant and powerful tools to make short work of this. Let's suppose that we have some code for a game library:

.. code-block:: python
   :caption: Contents of ``game_library.py``
   
   # Some code that we want to make configurable

   class Character:
       def __init__(self, name: str, level: int = 1, inventory=None):
           self.name = name
           self.level = level
           self.inventory = inventory
 
       def __repr__(self):
           return f"{self.name}, lvl: {self.level}, has: {self.inventory}"


   def inventory(gold: int, costume: str) -> dict:
       return {"gold": gold, "costume": costume}

There are two steps that we need to take to design our application using hydra-zen and Hydra.
 
1. Use hydra-zen's functions to automatically generate configurable interfaces to our library code.
2. Design a task function that instantiates these configs and uses them to perform our application's desired functionality.

Let's design an application where we can configure ``game_library.Character`` from a command line interface (CLI), including the items in its nested inventory.

.. code-block:: python
   :caption: Basic usage of hydra-zen

   # contents of my_app.py
   import hydra
   from hydra.core.config_store import ConfigStore
   from hydra_zen import builds, instantiate, make_config
   
   from game_library import inventory, Character
   
   # Using `hydra_zen.builds` to generate configs for our library's code, 
   # based on the signatures of the objects
   InventoryConf = builds(inventory, populate_full_signature=True)
   CharConf = builds(Character, inventory=InventoryConf, populate_full_signature=True)
    
   # Registering `CharConf` as the top-level config for our application
   cs = ConfigStore.instance()
   cs.store(name="char_conf", node=CharConf)
   
   # `@hydra.main` creates a CLI for our application based on `CharConf`
   @hydra.main(config_path=None, config_name="char_conf")
   def task_function(cfg: CharConf):
       # Performs a task using our populated configurations

       # `instantiate(cfg)` instantiates `Character` using the values
       # that we specify at the command line 
       player = instantiate(cfg)   
       
       assert isinstance(player, Character)
       print(player)
       return player


   if __name__ == "__main__":
       task_function()

We can now configure and launch our application from the command line!

.. code-block:: console
   :caption: Configuring and launching our application.

   $ python my_app.py name=Bob inventory.gold=10 inventory.costume=robe
   Bob, lvl: 1, has: {'gold': 10, 'costume': 'robe'}
   
   $ python my_app.py name=Jade level=44 inventory.gold=0 inventory.costume=pajamas
   Jade, lvl: 44, has: {'gold': 0, 'costume': 'pajamas'}

This example just scratches the surface of what can be achieved using hydra-zen and Hydra! Please work through our :ref:`tutorials <tutorials>` to gain a deeper understanding of these tools. You will see that the application is not only highly **configurable**, but also **reproducible** and **scalable**.


What is the Difference Between Hydra and hydra-zen?
===================================================

.. admonition:: The Answer
   
   **Hydra** provides a framework for configuring and launching applications. **hydra-zen** provides elegant tools for writing and validating configs for the Hydra framework, and for launching applications in the Hydra framework.

Hydra provides a powerful framework for configuring and launching software applications 
and a reproducible and scalable manner. It does not, however, provide tooling for 
actually writing the configs for your application; Hydra users typically write, by hand, yamls or dataclasses for all of their configs. For the example above one would typically write:

.. code-block:: python
   :caption: Writing configs *without* hydra-zen
   
   from dataclasses import dataclass
   from typing import Any
   
   @dataclass
   class InventoryConf:
       gold: int
       costume: str
       _target_: str = "game_library.Character"
           
   @dataclass
   class CharConf:
       name: str
       level: int = 1
       inventory: Any = InventoryConf
       _target_: str = "game_library.Character"


In contrast to this, hydra-zen provides specialized tooling for creating configs for use with the Hydra framework. Thus it is designed to be used *with* Hydra, and not instead of Hydra. Rather than writing the above dataclass-based configs, we can simply use :func:`~hydra_zen.builds` to generate the equivalent dataclasses:

.. code-block:: python
   :caption: Generating configs with hydra-zen
   
   from hydra_zen import builds
   from game_library import inventory, Character

   InventoryConf = builds(inventory, populate_full_signature=True)
   CharConf = builds(Character, inventory=InventoryConf, populate_full_signature=True)

Beyond saving us from having to :ref:`repeat ourselves in our code <dry>`, there are many additional benefits from using hydra-zen's functionality, some of which are listed at the top of this page. 

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
