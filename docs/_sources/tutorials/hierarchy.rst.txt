.. admonition:: Prerequisites

   This tutorial assumes that you have completed the earlier tutorials: :ref:`basic-app`
   and :ref:`cli-app`.

.. _hierarchy-app:

===================================
Design a Hierarchical App Interface
===================================

In this tutorial we will design an app that has an interface that is hierarchical in
nature. This particular app will describe a player in a video game; this player has a 
configurable name and experience-level, as well as an inventory, which itself has 
configurable components.

.. _game-library:

Creating (Fake) Library Code
============================

Often times the interface of an app is determined by existing classes and functions in 
our library's code. Let's create a new Python script, ``game_library.py``, in the same
directory as ``my_app.py``. This will serve as a mimic of a "real" Python library.

In this script we'll define a :class:`Character` class and an :func:`inventory` 
function as follows. Populate ``game_library.py`` with the following code.

.. code-block:: python
   :caption: Contents of ``game_library.py``
   
   # Should be in same directory as `my_app.py`
   
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

.. note::

   :plymi:`Type-annotations <Module5_OddsAndEnds/Writing_Good_Code.html#Type-Hinting>` are **not** required by hydra-zen. However, they do enable :ref:`runtime type-checking of configured values <type-support>` for our app.


To see this code in action, open a Python console (or Jupyter notebook) in the same 
directory as ``game_library.py`` and reproduce the following steps.

.. code-block:: pycon
   :caption: Getting a feel for the code in ``game_library.py``

   >>> from game_library import Character, inventory
   >>> stuff = inventory(gold=12, weapon="stick", costume="bball jersey")
   
   >>> Character("bowser", inventory=stuff)
   bowser, lvl: 1, has: {'gold': 12, 'weapon': 'stick', 'costume': 'bball jersey'}


Modifying Our App
=================

Let's change our app so that the interface describes only one player, instead of two.
We want to be able to configure the player based on the following hierarchy of fields:

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

Because configurable aspects of our app should directly reflect the interfaces of 
:class:`Character` class and :func:`inventory`, we can use
:func:`~hydra_zen.builds` to generate configs that reflect these interfaces. 

To see :func:`~hydra_zen.builds` in action, open a Python console (or Jupyter notebook) in the same directory as ``game_library.py``. Follow along with these inputs.

.. code-block:: pycon
   :caption: Getting a feel for :func:`~hydra_zen.builds`

   >>> from hydra_zen import builds, instantiate, to_yaml
   >>> from game_library import Character
   
   >>> CharConf = builds(Character, populate_full_signature=True)
   >>> print(to_yaml(CharConf))
   _target_: game_library.Character
   name: ???
   level: 1
   inventory: null
   
   >>> instantiate(CharConf, name="celeste")
   celeste, lvl: 1, has: None

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

Finally, the top-level config for our app will simply specify that ``player`` is described by this character config:

.. code-block:: python
   :caption: The top-level config for our app

   from hydra_zen import make_config

   Config = make_config(player=CharConf)


Updating the Task Function
--------------------------

We'll make some trivial modifications to our task function. We're only dealing with one 
player now, not two, so we adjust accordingly. Let's also print the 
``Character``-instance for ``player`` so that we get instant feedback as we prototype our app.

.. code-block:: python
   :caption: A revised task function (single-player only)

   def task_function(cfg: Config):
       obj = instantiate(cfg)
       
       player = obj.player
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
    cs.store(name="my_app", node=Config)
    
    
    @hydra.main(config_path=None, config_name="my_app")
    def task_function(cfg: Config):
        obj = instantiate(cfg)
        
        player = obj.player
        print(player)
        
        with open("player_log.txt", "w") as f:
            f.write("Game session log:\n")
            f.write(f"Player: {player}\n")

        return player
    
    if __name__ == "__main__":
        task_function()


Running Our App
===============

We can now configure any aspect of the player when launching our app; let's try a few 
examples in order to get a feel for the syntax. 

Open your terminal in the directory shared by both ``my_app.py`` and 
``game_library.py`` and run the following commands. Verify that you can reproduce the 
behavior shown below.

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
    
   We can use :func:`hydra_zen.launch` to launch our app, instead of using our app's 
   CLI. The following command line expression

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

To inspect the most-recent log written by our app, let's open a Python terminal in the same directory as ``my_app.py`` and define the following function for reading files

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
   
   >>> print_file(latest_job / ".hydra" / "overrides.yaml")
   - player.name=frodo
   - player.level=2
   - player.inventory.costume=robe

Great! Our app is now much more sophisticated: its configurable interface reflects - 
dynamically - the library code that we are ultimately instantiating. We also see the 
power of Hydra's ability to configure nested fields within our config.

In the next tutorial, we will define swappable config groups so that we can load 
specific player profiles and inventory load-outs from our app's interface.

.. admonition:: References

   - `~hydra_zen.make_custom_builds_fn`
   - `~hydra_zen.builds`
   - :hydra:`Hydra's Config Store API <tutorials/structured_config/config_store>`
   - :hydra:`Hydra's command line override syntax <advanced/override_grammar/basic>`

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our app.

