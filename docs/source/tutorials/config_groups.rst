===========================
Add Swappable Config Groups
===========================

.. admonition:: Prerequisites

   This tutorial is a direct follow-on to: :ref:`hierarchy-app`.

In this tutorial we will create swappable configuration "groups" for our app; this will 
enable us to create specific player profiles and inventory load-outs. These config 
groups can then be specified by-name when we launch our app.

Modifying Our App
=================

We will need to make three modifications to our code in ``my_app.py``:

1. Create additional inventory configs and player-profile configs.
2. Register these configs - by name - in Hydra's config store, under the appropriate respective groups.
3. Update our app's top-level config to include a "defaults" list, which is a special field that Hydra uses to define the interface to our app.


Creating Inventory-Config Groups
--------------------------------

In the previous tutorial, we created a config that represented an inventory of "starter gear". Let's simply create some other inventory "load-outs" that we can pick from. We 
will use the following code in ``my_app.py``.

.. attention:: 

   The ``game_library`` code used below comes from a script that we, the 
   tutorial-readers, created ourselves. See :ref:`game-library` for details.

.. code-block:: python
   :caption: Creating configs for new inventory load-outs

   from hydra_zen import make_custom_builds_fn

   from game_library import inventory

   builds = make_custom_builds_fn(populate_full_signature=True)

   InventoryConf = builds(inventory)
   starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
   advanced_gear = InventoryConf(gold=500, weapon="wand", costume="magic robe")
   hard_mode_gear = InventoryConf(gold=0, weapon="inner thoughts", costume="rags")

Now we will register each of these configs in Hydra's config store. Each of these will 
be stored with a distinct name, but under the same group: ``player/inventory``.

.. code-block:: python
   :caption: Adding configs to the ``player/inventory`` group in Hydra's config store

   from hydra.core.config_store import ConfigStore


   cs = ConfigStore.instance()

   cs.store(group="player/inventory", name="starter", node=starter_gear)
   cs.store(group="player/inventory", name="advanced", node=advanced_gear)
   cs.store(group="player/inventory", name="hard_mode", node=hard_mode_gear)


Creating Player-Config Groups
-----------------------------

Now, let's imagine that we have a couple of players who have saved their player 
profiles, so that they can resume progress when they play our game. Let's mock-up these 
player configs and then add them to the config store.

.. code-block:: python
   :caption: Creating player-profile configs
   
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

We will add these to Hydra's config store under the ``player`` group, so that these 
particular player-profiles can be used by-name when we launch our app.


.. code-block:: python
   :caption: Adding configs to the ``player`` group in Hydra's config store

   cs.store(group="player", name="base", node=CharConf)
   cs.store(group="player", name="brinda", node=brinda_conf)
   cs.store(group="player", name="rakesh", node=rakesh_conf)


Updating Our Top-Level Config 
-----------------------------

With these groups specified, we can tell Hydra to use a particular group-entry as a 
default config for that group. Let's specify the ``CharConf`` config, which we named ``base`` in the config store, as the default player-profile.


.. code-block:: python
   :caption: Adding configs to the ``player`` group in Hydra's config store

   Config = make_config("player", defaults=["_self_", {"player": "base"}])
   cs.store(name="config", node=Config)


Putting It All Together
-----------------------

Let's update the contents of ``my_app.py`` to reflect the changes that we just went 
over. Modify your ``my_app.py`` script to match the following code.

.. code-block:: python
   :caption: Contents of ``my_app.py``

   import hydra
   from hydra.core.config_store import ConfigStore
   
   from hydra_zen import instantiate, make_config, make_custom_builds_fn
   
   from game_library import inventory, Character
   
   builds = make_custom_builds_fn(populate_full_signature=True)
   
   cs = ConfigStore.instance()

   # Create inventory configs
   InventoryConf = builds(inventory)
   starter_gear = InventoryConf(gold=10, weapon="stick", costume="tunic")
   advanced_gear = InventoryConf(gold=500, weapon="wand", costume="magic robe")
   hard_mode_gear = InventoryConf(gold=0, weapon="inner thoughts", costume="rags")
   
   # Register inventory configs under group: player/inventory
   cs.store(group="player/inventory", name="starter", node=starter_gear)
   cs.store(group="player/inventory", name="advanced", node=advanced_gear)
   cs.store(group="player/inventory", name="hard_mode", node=hard_mode_gear)
   
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
   cs.store(group="player", name="base", node=CharConf)
   cs.store(group="player", name="brinda", node=brinda_conf)
   cs.store(group="player", name="rakesh", node=rakesh_conf)
   
   # Specify default group for player to be: base
   Config = make_config("player", defaults=["_self_", {"player": "base"}])
   
   cs.store(name="config", node=Config)
   
   
   @hydra.main(config_path=None, config_name="config")
   def task_function(cfg: Config):
       cfg = instantiate(cfg)
   
       player = cfg.player
       print(player)
   
       with open("player_log.txt", "w") as f:
           f.write("Game session log:\n")
           f.write(f"Player: {player}\n")
   
       return player
   
   
   if __name__ == "__main__":
       task_function()


Running Our App
===============

In addition to configuring any aspect of the player manually, we can now also reference particular config-group items by-name when we launch our app.

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



.. admonition:: References

   - `~hydra_zen.make_custom_builds_fn`
   - `~hydra_zen.builds`
   - :hydra:`Hydra's Config Store API <tutorials/structured_config/config_store>`
   - :hydra:`Hydra's command line override syntax <advanced/override_grammar/basic>`

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our app.

