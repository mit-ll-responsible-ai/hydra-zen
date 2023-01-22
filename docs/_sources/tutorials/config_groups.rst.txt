.. meta::
   :description: A tutorial that adds configuration groups to a Hydra project.

.. admonition:: Prerequisites

   This tutorial is a direct follow-on to: :ref:`hierarchy-app`.

.. _config-groups-tutorial:

======================================
Provide Swappable Configuration Groups
======================================

In this tutorial we will create swappable configuration groups for our application: we 
will pre-configure specific player profiles and inventory load-outs. By storing these 
configs in a config store, we will be able to load them and swap them out by group & 
name from the command line.

Modifying Our Application
=========================

We will need to make three modifications to our code in ``my_app.py``:

1. Create additional inventory configs and player-profile configs.
2. Register these configs - by group and name - in a config store.
3. Update our application's top-level config to include a "Hydra defaults" list, which is a special field that Hydra uses to define the default inputs to our application.


Creating Inventory-Config Groups
--------------------------------

In the previous tutorial, we created a config that represented an inventory of "starter gear". Let's simply create some other inventory "load-outs" that we can pick from. We 
will use the following code in ``my_app.py``.

.. attention:: 

   The ``game_library`` code used below comes from a script that we, the 
   tutorial-readers, created ourselves. See :ref:`game-library` for details.

.. code-block:: python
   :caption: Creating configs for new inventory load-outs.

   from hydra_zen import make_custom_builds_fn

   from game_library import inventory

   builds = make_custom_builds_fn(populate_full_signature=True)

   InventoryConf = builds(inventory)
   starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
   advanced_gear = InventoryConf(gold=500, weapon="wand", costume="magic robe")
   hard_mode_gear = InventoryConf(gold=0, weapon="inner thoughts", costume="rags")

Now we will register each of these configs using :class:`~hydra_zen.ZenStore`. Each of 
these will be stored with a distinct name, but under the same group: 
``player/inventory``.

.. code-block:: python
   :caption: Adding configs to the ``player/inventory`` group in hydra-zen's config store.

   from hydra_zen import store

   # pre-set the group name
   inv_store = store(group="player/inventory")

   inv_store(starter_gear, name="starter")
   inv_store(advanced_gear, name="advanced")
   inv_store(hard_mode_gear, name="hard_mode")

Creating Player-Config Groups
-----------------------------

Suppose that we have a couple of players who have saved their player profiles, so that 
they can resume progress when they play our game. Let's mock-up these player configs 
and then add them to the config store.

.. code-block:: python
   :caption: Creating player-profile configs.
   
   from game_library import Character

   CharConf = builds(Character, inventory=starter_gear)
   
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

We will add these to the same config store, but under the ``player`` group. This will 
enable us to load these particular player-profiles by-name when we launch our 
application.


.. code-block:: python
   :caption: Adding configs to the ``player`` group in hydra-zen's config store.

   # pre-set the group name
   player_store = store(group="player")

   player_store(CharConf, name="base")
   player_store(brinda_conf, name="brinda")
   player_store(rakesh_conf, name="rakesh")


Updating Our Top-Level Config 
-----------------------------

With these groups specified, we can tell Hydra to use a particular group-entry as a 
default config for that group by specifying a `hydra_defaults` in our task function's 
config. For this example, let's specify the ``CharConf`` config, which we named 
``base`` in the config store, as the default player-profile.


.. code-block:: python
   :caption: Specifying the player-group item named ``base`` as the default player-profile.

   @store(name="my_app",  hydra_defaults=["_self_", {"player": "base"}])
   def task_function(player: Character):
       ...

.. note:: 

   The ``hydra-defaults`` field in our top-level config has special meaning in the 
   context of Hydra: it specifies a list that instructs Hydra how to build the 
   resulting config, and the list itself is not included in the config. You can read 
   about the Defaults List in 
   :hydra:`this tutorial <tutorials/structured_config/defaults>` and in this :hydra:`technical reference <advanced/defaults_list>`.


Putting It All Together
-----------------------

Let's update the contents of ``my_app.py`` to reflect the changes that we just went 
over. Modify your ``my_app.py`` script to match the following code.

.. code-block:: python
   :caption: Contents of ``my_app.py``

   from hydra_zen import store, make_custom_builds_fn, zen
   
   from game_library import inventory, Character
   
   builds = make_custom_builds_fn(populate_full_signature=True)
   
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
   
   # Create player-profile configs
   CharConf = builds(Character, inventory=starter_gear)
   
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

.. tip::

   **A matter of housekeeping**: our configs need not be defined in the same file as
   ``task_function``. They can be defined - and added to the config store - in a 
   separate file in our library, e.g. ``configs.py``, or across multiple files. This is 
   nice from an organizational perspective, plus it enables to use these configs
   across multiple applications.


Running Our Application
=======================

In addition to configuring any aspect of the player manually, we can now also reference particular config-group items by-name when we launch our application.

Open your terminal in the directory shared by both ``my_app.py`` and 
``game_library.py`` and run the following commands.
The ``--help`` flag will list our application's configurable groups and hierarchical
parameters:

.. code-block:: console
   :caption: Viewing the ``--help`` info for our application.

   $ python my_app.py --help
   my_app is powered by Hydra.
   
   == Configuration groups ==
   Compose your configuration from those groups (group=option)
   
   player: base, brinda, rakesh
   player/inventory: advanced, hard_mode, starter
   
   
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


Verify that you can reproduce the behavior shown below.

.. code-block:: console
   :caption: Default inventory.

   $ python my_app.py player.name=ivy
   ivy, lvl: 1, has: {'gold': 10, 'weapon': 'stick', 'costume': 'tunic'}

.. code-block:: console
   :caption: Give player 'hard-mode' load-out.

   $ python my_app.py player.name=ivy +player/inventory=hard_mode
   ivy, lvl: 1, has: {'gold': 0, 'weapon': 'inner thoughts', 'costume': 'rags'}

.. code-block:: console
   :caption: Player-level 3. With 'hard-mode' load-out, but with 10 gold.

   $ python my_app.py player.name=ivy player.level=3 +player/inventory=hard_mode player.inventory.gold=10
   ivy, lvl: 3, has: {'gold': 10, 'weapon': 'inner thoughts', 'costume': 'rags'}

.. code-block:: console
   :caption: Load Rakesh's player-profile

   $ python my_app.py player=rakesh
   rakesh, lvl: 300, has: {'gold': 41, 'weapon': 'pillow', 'costume': 'PJs'}

.. code-block:: console
   :caption: Load Brinda's player-profile, and change their costume

   $ python my_app.py player=brinda player.inventory.costume=armor
   brinda, lvl: 47, has: {'gold': 52, 'weapon': 'flute', 'costume': 'armor'}


Wonderful! Using config groups in our app makes it trivial to swap-out entire "modules" 
of our app's config. This is an elegant way to change, en-masse, pieces of functionality that are being used by our app.

In the final section of this tutorial, we will use hydra-zen to "inject" novel 
functionality into our code without having to modify our library's source code nor our task function.

Reference Documentation
=======================
Want a deeper understanding of how hydra-zen and Hydra work?
The following reference materials are especially relevant to this
tutorial section.

- :hydra:`Hydra's default list <tutorials/structured_config/defaults>`
- :hydra:`Hydra's default list (technical reference) <advanced/defaults_list>`
- :hydra:`Hydra's Config Store API <tutorials/structured_config/config_store>`
- :hydra:`Hydra's command line override syntax <advanced/override_grammar/basic>`
- `~hydra_zen.make_custom_builds_fn`
- `~hydra_zen.builds`
- `~hydra_zen.instantiate`

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our application.
