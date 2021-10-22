===================================
Design a Hierarchical App Interface
===================================

In this tutorial we will design an app that has an interface that is hierarchical in
nature. This particular app will describe a player in a video game; this player has a 
configurable name and experience-level, as well as an inventory, which itself has 
configurable components.

.. admonition:: Prerequisites

   This tutorial assumes that you have completed the earlier tutorials: :ref:`basic-app`
   and :ref:`cli-app`.

Creating (Fake) Library Code
============================

Often times the interface of an app is determined by existing classes and functions in 
our library's code. Let's create a new Python script, ``game_library.py``, in the same
directory as ``my_app.py``. This will serve as a mimic of a "real" Python library.

In this script we'll define a :class:`Character` class and an :func:`inventory` 
function as follows.

.. code-block:: python
   :caption: Contents of game_library.py:
   
   # Note: type annotations are *not* required by hydra-zen

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

To see this code in action, open a Python console (or Jupyter notebook) in the same 
directory as ``game_library.py`` and reproduce the following steps.

.. code-block:: pycon

   >>> from game_library import Character, inventory
   >>> stuff = inventory(gold=12, weapon="stick", costume="bball jersey")
   
   >>> Character("bowser", inventory=stuff)
   bowser, lvl: 1, has: {'gold': 12, 'weapon': 'stick', 'costume': 'bball jersey'}


Modifying Our App
=================

Let's change our app so that the interface describes only one player, instead of two.
We want to be able to configure the player's

- name
- level
- gold in their inventory
- weapon in their inventory
- costume in their inventory

Dynamically Generating Configs
------------------------------

The configurable aspects of our app is already reflected in the interfaces of 
:class:`Character` class and :func:`inventory`. Thus we can use 
:func:`~hydra_zen.builds` to generate configs based on these interfaces. We'll
create a configuration for a character with basic "starter gear" for their 
inventory.

.. code-block:: python

   from hydra_zen import make_custom_builds_fn 
   
   from game_library import inventory, Character

   builds = make_custom_builds_fn(populate_full_signature=True)

   InventoryConf = builds(inventory)
   starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
   
   CharConf = builds(Character, inventory=starter_gear)

Then the config for our app will simply specify that ``player`` is described by this 
character config:

.. code-block:: python

   from hydra_zen import make_config

   Config = make_config(player=CharConf)


Updating the Task Function
--------------------------

We'll make some trivial modifications to our task function. We're only dealing with one 
player now, not two, so we adjust accordingly. Let's also print the 
``Character``-instance for ``player`` so that it is easy for us to get instant feedback 
from the app.

.. code-block:: python

   def task_function(cfg: Config):
       cfg = instantiate(cfg)
       
       player = cfg.player
       print(player)

       with open("player_log.txt", "w") as f:
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

    import hydra
    from hydra.core.config_store import ConfigStore
    
    from hydra_zen import instantiate, make_config, make_custom_builds_fn
    
    from game_library import inventory, Character
    
    builds = make_custom_builds_fn(populate_full_signature=True)
    
    # generating configs
    InventoryConf = builds(inventory)
    starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
    
    CharConf = builds(Character, inventory=starter_gear)
    
    # creating the top-level config for our app
    Config = make_config(player=CharConf)
    
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    
    
    @hydra.main(config_path=None, config_name="config")
    def task_function(cfg: Config):
        cfg = instantiate(cfg)
        player = cfg.player
        print(player)
        with open("player_log.txt", "w") as f:
            f.write("Game session log:\n")
            f.write(f"Player: {player}\n")
    
    
    if __name__ == "__main__":
        task_function()


Running Our App
===============

We can configure any aspect of the player when launching our app; let's try a few 
examples in order to get a feel for the syntax

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



Inspecting the Results
----------------------

To inspect the log written by our app, let's open a Python terminal in the same 
directory as ``my_app.py`` and define the following function for reading files

.. code-block:: pycon

   >>> from pathlib import Path 
   >>> def print_file(x: Path):
   ...     with x.open("r") as f: 
   ...         print(f.read())


Getting the directory containing the output of this job:

.. code-block:: pycon
   
   >>> *_, latest_job = sorted((Path.cwd() / "outputs").glob("*/*"))
   >>> latest_job  # changes based  on reader's date, time, and OS
   WindowsPath('C:/outputs/2021-10-22/00-19-52')

Let's check the log file that our app wrote. ``player_log.txt`` should read as follows.

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


We can also check to see what the exact "overrides" that were used to launch the app 
for this job in ``.hydra/overrides.yaml``.

.. code-block:: pycon
   
   >>> print_file(latest_job / ".hydra" / "config.yaml")
   - player.name=frodo
   - player.level=2
   - player.inventory.costume=robe

.. admonition:: References

   Refer to :func:`~hydra_zen.make_config` for more details about designing configs, including creating configs with default  values, and with type-annotations for type-checking.

   Refer to :func:`~hydra_zen.launch` to learn more about the ``JobReturn`` object that
   is produced by our job, and to see an app run in a multirun fashion.

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our app.

