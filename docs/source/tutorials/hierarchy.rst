.. admonition:: Prerequisites

   This tutorial assumes that you have completed the earlier tutorials: :ref:`basic-app`
   and :ref:`cli-app`.

.. _hierarchy-app:

==================================================
Design a Hierarchical Interface for an Application
==================================================

In this tutorial we will design an application that has an interface that is 
hierarchical in nature. This particular application will describe a player in a video 
game; this player has a configurable name and experience-level, as well as an 
inventory, which itself has configurable components.

.. _game-library:

Creating (Fake) Library Code
============================

Often times the interface of an application is determined by existing classes and 
functions in our library's code. Let's create a new Python script, ``game_library.py``, 
in the same directory as ``my_app.py``. This will serve as a mimic of a "real" Python 
library.

In this script we'll define a :class:`Character` class and an :func:`inventory` 
function as follows. Populate ``game_library.py`` with the following code.

.. code-block:: python
   :caption: Contents of ``game_library.py``
   
   # Should be in same directory as `my_app.py`
   
   # Note: type annotations are *not* required by hydra-zen

   __all__ = ["inventory", "Character"]


   class Character:
       def __init__(self, name: str, level: int = 1, inventory=None):
           self.name = name
           self.level = level
           self.inventory = inventory
 
       def __repr__(self):
           out = ""
           out += f"{self.name}, "
           out += f"lvl: {self.level}, "
           out += f"has: {self.inventory}"
           return out


   def inventory(gold: int, weapon: str, costume: str):
       return {"gold": gold, "weapon": weapon, "costume": costume}

.. note::

   :plymi:`Type-annotations <Module5_OddsAndEnds/Writing_Good_Code.html#Type-Hinting>` are **not** required by hydra-zen. However, they do enable :ref:`runtime type-checking of configured values <type-support>` for our application.


To see this code in action, open a Python console (or Jupyter notebook) in the same 
directory as ``game_library.py`` and reproduce the following steps.

.. code-block:: pycon
   :caption: Getting a feel for the code in ``game_library.py``

   >>> from game_library import Character, inventory
   >>> stuff = inventory(gold=12, weapon="stick", costume="bball jersey")
   
   >>> Character("bowser", inventory=stuff)
   bowser, lvl: 1, has: {'gold': 12, 'weapon': 'stick', 'costume': 'bball jersey'}


Modifying Our Application
=========================

Let's change our application so that the interface describes only one player, instead 
of two. We want to be able to configure the player based on the following hierarchy of 
fields:

**Player**

- name
- level
- inventory
   
  * amount of gold
  * weapon type
  * costume

These fields reflect the interfaces/structure of :class:`Character` and 
:func:`inventory`.

Dynamically Generating Configs
------------------------------

Because configurable aspects of our application should directly reflect the interfaces 
of :class:`Character` class and :func:`inventory`, we can use :func:`~hydra_zen.builds` 
to generate configs that reflect these interfaces. 

To see :func:`~hydra_zen.builds` in action, open a Python console (or Jupyter notebook) in the same directory as ``game_library.py``. Follow along with these inputs.

.. code-block:: pycon
   :caption: Getting a feel for :func:`~hydra_zen.builds`
   
   >>> from hydra_zen import builds, instantiate, to_yaml
   >>> from game_library import Character
   
   >>> def print_yaml(x): print(to_yaml(x))
   
   >>> CharConf = builds(Character, populate_full_signature=True)
   
   >>> print_yaml(CharConf)
   _target_: game_library.Character
   name: ???
   level: 1
   inventory: null
   
   >>> print_yaml(CharConf(name="celeste"))
   _target_: game_library.Character
   name: celeste
   level: 1
   inventory: null

The :func:`~hydra_zen.instantiate` function is used to actually "build" the object described by our config

.. code-block:: pycon
   :caption: Getting a feel for  :func:`~hydra_zen.instantiate`.

   >>> from hydra_zen import instantiate
   
   >>> char = instantiate(CharConf(name="celeste"))
   
   >>> char
   celeste, lvl: 1, has: None

   >>> isinstance(char, Character)
   True

Let's create a configuration for a character with basic "starter gear" for their 
inventory. We will use the following code in ``my_app.py``.

.. code-block:: python
   :caption: Dynamically generating configs based on ``game_library``

   from hydra_zen import make_custom_builds_fn 
   
   from game_library import inventory, Character

   builds = make_custom_builds_fn(populate_full_signature=True)

   InventoryConf = builds(inventory)
   starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
   
   # note: 
   # We are nesting the config for `inventory` within the 
   # config for `Character`.
   CharConf = builds(Character, inventory=starter_gear)


Updating the Task Function
--------------------------

We'll make some modifications to our task function.

- We're only dealing with one player now, not two, so we adjust accordingly.
- Let's print ``Character``-instance for ``player`` so that we get instant feedback when we run our application from the CLI.

.. code-block:: python
   :caption: A revised task function (single-player only)

   def task_function(player: Character):

      print(player)

      with open("player_log.txt", "a") as f:
         f.write("Game session log:\n")
         f.write(f"Player: {player}\n")

      return player


Piecing It All Together
-----------------------

Combining these configs and task function together - along with the boilerplate code 
needed to :ref:`create a command line interface <cli-app>` - our updated ``my_app.py`` 
script is as follows.

.. code-block:: python
   :caption: Contents of my_app.py:

   from hydra_zen import make_custom_builds_fn, store, zen
   
   from game_library import inventory, Character
   
   builds = make_custom_builds_fn(populate_full_signature=True)
   
   # generating configs
   starter_gear = builds(inventory, gold=10, weapon="stick", costume="tunic")
   
   CharConf = builds(Character, inventory=starter_gear)
   
   # Generate and store a top-level config specifying `CharConf` as the
   # default config for `player`
   @store(name="my_app", player=CharConf)
   def task_function(player: Character):
   
       print(player)
   
       with open("player_log.txt", "a") as f:
           f.write("Game session log:\n")
           f.write(f"Player: {player}\n")
   
       return player
   
   if __name__ == "__main__":
       store.add_to_hydra_store()
       zen(task_function).hydra_main(config_name="my_app", 
                                     version_base="1.1",
                                     config_path=".",
                                     )


Running Our Application
=======================

We can now configure any aspect of the player when launching our application; let's try 
a few examples in order to get a feel for the syntax. 
Open your terminal in the directory shared by both ``my_app.py`` and 
``game_library.py`` and run the following commands. Verify that you can reproduce the 
behavior shown below.

Checking the ``--help`` option of our application reveals the hierarchical structure of 
its configurable interface. See that the only required value is ``player.name``,  and
that we can override any of the other default configured values.

.. code-block:: console
   :caption: Checking the configurable components of our app. (We will add configuration groups in a later lesson.)

   $ python my_app.py --help
   my_app is powered by Hydra.
   
   == Configuration groups ==
   Compose your configuration from those groups (group=option)
   
   
   
   == Config ==
   Override anything in the config (foo.bar=value)
   
   player:
     _target_: game_library.Character
     name: ???
     level: 1
     inventory:
       _target_: game_library.inventory
       gold: 10
       weapon: stick
       costume: tunic


Now let's run our application with various configurations.

.. code-block:: console
   :caption: Configuring: name

   $ python my_app.py player.name=frodo
   frodo, lvl: 1, has: {'gold': 10, 'weapon': 'stick', 'costume': 'tunic'}

.. code-block:: console
   :caption: Configuring: name and level

   $ python my_app.py player.name=frodo player.level=5
   frodo, lvl: 5, has: {'gold': 10, 'weapon': 'stick', 'costume': 'tunic'}

.. code-block:: console
   :caption: Configuring: name, level, and costume

   $ python my_app.py player.name=frodo player.level=2 player.inventory.costume=robe
   frodo, lvl: 2, has: {'gold': 10, 'weapon': 'stick', 'costume': 'robe'}

.. note:: 
    
   We can use :func:`hydra_zen.launch` to launch our application, instead of using our 
   application's CLI. The following command line expression

   .. code-block:: console
   
      $ python my_app.py player.name=frodo player.level=2 player.inventory.costume=robe
      frodo, lvl: 2, has: {'gold': 10, 'weapon': 'stick', 'costume': 'robe'}

   can be replicated from a Python console via:

   .. code-block:: pycon
      :caption: A Python console, opened in the same directory as ``my_app.py``
      
      >>> from hydra_zen import launch
      >>> from my_app import Config, task_function
      >>> job = launch(
      ...     Config,
      ...     task_function,
      ...     ["player.name=frodo", "player.level=2", "player.inventory.costume=robe"],
      ... )
      frodo, lvl: 2, has: {'gold': 10, 'weapon': 'stick', 'costume': 'robe'}


Inspecting the Results
----------------------

To inspect the most-recent log written by our application, let's open a Python terminal 
in the same directory as ``my_app.py`` and define the following function for reading 
files

.. code-block:: pycon

   >>> from pathlib import Path 
   >>> def print_file(x: Path):
   ...     with x.open("r") as f: 
   ...         print(f.read())


Getting the directory containing the output of the most-recent job:

.. code-block:: pycon
   
   >>> *_, latest_job = sorted((Path.cwd() / "outputs").glob("*/*"))
   >>> latest_job  # changes based  on reader's date, time, and OS
   WindowsPath('C:/outputs/2021-10-22/00-19-52')

Let's check the log file that our application wrote. ``player_log.txt`` should read as 
follows.

.. code-block:: pycon
   
   >>> print_file(latest_job / "player_log.txt")
   Game session log:
   Player: frodo, lvl: 2, has: {'gold': 10, 'weapon': 'stick', 'costume': 'robe'}

Hydra details the hierarchical config passed to our task function; let's look at the 
contents of ``.hydra/config.yaml``.

.. code-block:: pycon
   
   >>> print_file(latest_job / ".hydra" / "config.yaml")
   player:
     _target_: game_library.Character
     name: frodo
     level: 2
     inventory:
       _target_: game_library.inventory
       gold: 10
       weapon: stick
       costume: robe


We can also check to see what the exact "overrides" that were used to launch the 
application for this job in ``.hydra/overrides.yaml``.

.. code-block:: pycon
   
   >>> print_file(latest_job / ".hydra" / "overrides.yaml")
   - player.name=frodo
   - player.level=2
   - player.inventory.costume=robe

Great! Our application is now much more sophisticated: its configurable interface 
reflects - dynamically - the library code that we are ultimately instantiating. We also 
see the power of Hydra's ability to configure nested fields within our config.

In the next tutorial, we will define swappable config groups so that we can load 
specific player profiles and inventory load-outs from our application's interface.

Reference Documentation
=======================
Want a deeper understanding of how hydra-zen and Hydra work?
The following reference materials are especially relevant to this
tutorial section.

- `~hydra_zen.instantiate`
- `~hydra_zen.builds`
- `~hydra_zen.make_custom_builds_fn`
- :hydra:`Hydra's Config Store API <tutorials/structured_config/config_store>`
- :hydra:`Hydra's command line override syntax <advanced/override_grammar/basic>`

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our application.
