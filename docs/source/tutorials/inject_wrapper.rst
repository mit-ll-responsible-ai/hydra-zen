.. meta::
   :description: A tutorial that demonstrates hydra-zen's function-injection mechanisms for instantiating structured configurations.

.. admonition:: Prerequisites

   This tutorial is a direct follow-on to: :ref:`config-groups-tutorial`.

=======================================================================
Inject Novel Functionality via the Application's Configurable Interface
=======================================================================

In this tutorial we will add new functionality into our application without modifying 
either our library's code or our task function. 

Let's suppose that the Halloween holiday is 
approaching, and we want to surprise our players with a treat: all of their costumes 
will display as "spooky" renditions during this season 🎃. We can use the 
"zen-wrappers" feature of `~hydra_zen.builds` to "inject" these updated costumes into 
the game, simply by modifying our application's config.

Modifying Our Application
=========================

We will need to make two modifications to our code in ``my_app.py``:

1. Define a wrapper that will modify the player's costume so that it is a "spooky" version of itself.
2. Include the wrapper in the player's config.


Creating the Functionality-Injecting Wrapper
--------------------------------------------

Let's create a `wrapper <https://realpython.com/primer-on-python-decorators/
#simple-decorators>`_ that will take in our :class:`game_library.Character` object, 
instantiate it, and then update its costume.

We will incorporate the following code in ``my_app.py``.

.. code-block:: python
   :caption: Defining the wrapper

   def halloween_update(CharClass):
       def wrapper(*args, **kwargs):
           # instantiate the Character
           char = CharClass(*args, **kwargs)
           
           costume = char.inventory["costume"]
           if costume:
               # update the costume
               char.inventory["costume"] = f"spooky {costume}"
           return char
       return wrapper

To see this in action, let's open a Python console (or Jupyter notebook) in the same 
directory as ``game_library.py`` and ``my_app.py``, and run the following.

.. attention:: 

   The ``game_library`` code used below comes from a script that we, the 
   tutorial-readers, created ourselves. See :ref:`game-library` for details.

.. code-block:: pycon
   :caption: Testing out the wrapper

   >>> from game_library import Character, inventory

   >>> def halloween_update(CharClass):
   ...     def wrapper(*args, **kwargs):
   ...         # instantiate the Character
   ...         char = CharClass(*args, **kwargs)
   ...         
   ...         costume = char.inventory["costume"]
   ...         if costume:
   ...             # update the costume
   ...             char.inventory["costume"] = f"spooky {costume}"
   ...         return char
   ...     return wrapper

   >>> WrappedChar = halloween_update(Character)  # character's costume will be made spooky

We see that the wrapped version of :class:`game_library.Character` automatically has 
its costume updated for the holiday. Verify that you see the following behaviors.

.. code-block:: pycon

   >>> WrappedChar(name="ness", inventory=inventory(gold=1, weapon="none", costume="shirt"))
   ness, lvl: 1, has: {'gold': 1, 'weapon': 'none', 'costume': 'spooky shirt'}
   
   >>> Character(name="ness", inventory=inventory(gold=1, weapon="none", costume="shirt"))
   ness, lvl: 1, has: {'gold': 1, 'weapon': 'none', 'costume': 'shirt'}


Including the Wrapper in Our Config
-----------------------------------

Incorporating this wrapper into our application simply involves specifying it as a 
"zen-wrapper" in our config for :class:`game_library.Character`. I.e. we will update:

.. code:: python 
   
   CharConf = builds(Character, ...)

to be

.. code:: python 
   
   CharConf = builds(Character, ..., zen_wrappers=halloween_update)


Putting It All Together
-----------------------

Let's update the contents of ``my_app.py`` to reflect the changes that we just went 
over. Modify your ``my_app.py`` script to match the following code.

.. code-block:: python
   :caption: Contents of ``my_app.py``

   from hydra_zen import store, make_custom_builds_fn, zen
   
   from game_library import inventory, Character
   
   builds = make_custom_builds_fn(populate_full_signature=True)
   
   
   # 1. Added our wrapper
   def halloween_update(CharClass):
       def wrapper(*args, **kwargs):
           # instantiate the Character
           char = CharClass(*args, **kwargs)
   
           costume = char.inventory["costume"]
           if costume:
               # update the costume
               char.inventory["costume"] = f"spooky {costume}"
           return char
   
       return wrapper
   

   # Create inventory configs
   InventoryConf = builds(inventory)
   starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
   advanced_gear = InventoryConf(gold=500, weapon="wand", costume="magic robe")
   hard_mode_gear = InventoryConf(gold=0, weapon="inner thoughts", costume="rags")
   
   # Register inventory configs under group: player/inventory
   inv_store = store(group="player/inventory")
   
   inv_store(starter_gear, name="starter")
   inv_store(advanced_gear, name="advanced")
   inv_store(hard_mode_gear, name="hard_mode")
   
   # 2. Included the wrapper in our config for `Character`
   CharConf = builds(Character, inventory=starter_gear, zen_wrappers=halloween_update)
   
   brinda_conf = CharConf(
       name="brinda",
       level=47,
       inventory=InventoryConf(costume="cape", weapon="flute", gold=52),
   )
   
   rakesh_conf = CharConf(
       name="rakesh",
       level=300,
       inventory=InventoryConf(costume="PJs", weapon="pillow", gold=41),
   )
   
   # Register player-profile configs under group: player
   player_store = store(group="player")

   player_store(CharConf, name="base")
   player_store(brinda_conf, name="brinda")
   player_store(rakesh_conf, name="rakesh")
   
   
   # The `hydra_defaults` field is specified in our task function's config.
   # It instructs Hydra to use the player config that named 'base' in our
   # config store as the default config for our app.
   @store(name="my_app",  hydra_defaults=["_self_", {"player": "base"}])
   def task_function(player: Character):
   
       print(player)
   
       with open("player_log.txt", "a") as f:
           f.write("Game session log:\n")
           f.write(f"Player: {player}\n")
   
       return player
   
   
   if __name__ == "__main__":
       # We need to add the configs from our local store to Hydra's
       # global config store
       store.add_to_hydra_store()
   
       # Our zen-wrapped task function is used to generate
       # the CLI, and to specify which config we want to use
       # to configure the app by default
       zen(task_function).hydra_main(config_name="my_app",
                                     version_base="1.1",
                                     config_path=".",
                                     )

Running Our Application
=======================

We can configure and launch our application exactly as we had before, but now all of 
the player's costumes will automatically become 👻 spooky 👻.

Open your terminal in the directory shared by both ``my_app.py`` and 
``game_library.py`` and run the following commands. Verify that you can reproduce the 
behavior shown below.

.. code-block:: console
   :caption: Base character.

   $ python my_app.py player.name=ivy
   ivy, lvl: 1, has: {'gold': 10, 'weapon': 'stick', 'costume': 'spooky tunic'}

.. code-block:: console
   :caption: Manually-specified costume.

   $ python my_app.py player.name=ivy player.inventory.costume=crown
   ivy, lvl: 1, has: {'gold': 10, 'weapon': 'stick', 'costume': 'spooky crown'}

.. code-block:: console
   :caption: Load Rakesh's player-profile

   $ python my_app.py player=rakesh
   rakesh, lvl: 300, has: {'gold': 41, 'weapon': 'pillow', 'costume': 'spooky PJs'}

Inspecting the Results
----------------------

Hydra will document our use of :func:`halloween_update` in the ``config.yaml`` for our 
job. To inspect the config for our most-recent job, let's open a Python terminal in the same directory as ``my_app.py`` and run the following code

.. code-block:: pycon

   >>> from pathlib import Path 
   >>> def print_file(x: Path):
   ...     with x.open("r") as f: 
   ...         print(f.read())  
   
   >>> *_, latest_job = sorted((Path.cwd() / "outputs").glob("*/*"))
   
   >>> print_file(latest_job / ".hydra" / "config.yaml")
   player:
     _target_: hydra_zen.funcs.zen_processing
     _zen_target: game_library.Character
     _zen_wrappers: __main__.halloween_update
     name: rakesh
     level: 300
     inventory:
       _target_: game_library.inventory
       gold: 41
       weapon: pillow
       costume: PJs

From this YAML config file, we can see explicitly that our application launched using 
the Halloween update; the update will also take effect if we were to re-launch our 
application using this particular YAML file to reproduce the job.

Outstanding! We successfully leveraged the zen-wrappers feature of
:func:`~hydra_zen.builds` to modify the behavior of our application, without touching 
our library's source code. And we did so in a self-documenting, and reproducible manner.

Although this achievement might not seem all that impressive in the context of this toy 
example, it should be emphasized that zen-wrappers can be used to inject arbitrary 
pre-processing, post-processing, and transformations into the config-instantiation 
process. For example, hydra-zen provides enhanced :ref:`data-validation capabilities <data-val>` via zen-wrappers. Based on this tutorial, we hope that you feel emboldened 
to design and use zen-wrappers in your workflow!

Reference Documentation
=======================
Want a deeper understanding of how hydra-zen and Hydra work?
The following reference materials are especially relevant to this
tutorial section.
   
- `~hydra_zen.builds`
- `Real Python's tutorial on wrappers (a.k.a decorators) <https://realpython.com/primer-on-python-decorators/#simple-decorators>`_

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our application.
