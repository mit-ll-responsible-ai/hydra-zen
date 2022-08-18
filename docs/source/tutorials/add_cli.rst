.. meta::
   :description: A tutorial for adding a command line interface to a Hydra program.

.. tip:: 
   Hover your cursor over any code block in this tutorial, and a clipboard will appear.
   Click it to copy the contents of the code block.

.. admonition:: Prerequisites

   This tutorial assumes that you have completed the earlier tutorial: :ref:`basic-app`

.. _cli-app:

===============================================
Add a Command Line Interface to Our Application
===============================================

In this tutorial we will update our project so that it can be configured and launched 
from a command line interface, using Hydra.


Modifying Our Project
=====================

Open ``my_app.py`` in your editor. We will make the following modifications to it:

1. Register our config in Hydra's "config store".
2. Decorate our task function to tell Hydra what config to use to define its interface.
3. Add a ``__main__`` clause to our ``my_app.py`` script so that the script runs our task function.

Modify your script to match this:

.. code-block:: python
   :caption: Contents of my_app.py:

   import hydra
   from hydra.core.config_store import ConfigStore
   
   from hydra_zen import instantiate, make_config
   
   Config = make_config("player1", "player2")
   
   # 1) Register our config with Hydra's config store
   cs = ConfigStore.instance()
   cs.store(name="my_app", node=Config)
   
   
   # 2) Tell Hydra what config to use for our task-function.
   #    The name specified here - 'config' - must match the
   #    name that we provided to `cs.store(name=<...>, node=Config)`
   @hydra.main(config_path=None, config_name="my_app")
   def task_function(cfg):
       # cfg: Config
       obj = instantiate(cfg)
       p1 = obj.player1
       p2 = obj.player2
   
       with open("player_log.txt", "w") as f:
           f.write("Game session log:\n")
           f.write(f"Player 1: {p1}\n" f"Player 2: {p2}")
   
   
   # 3) Executing `python my_app.py [...]` will run our task function
   if __name__ == "__main__":
       task_function()


Launching Our Application from the Command Line
===============================================

With the above modifications to ``my_app.py`` complete, we can launch our application 
from the command line. The following will launch a job with ``mario`` and ``luigi`` as 
the names for player 1 and player 2, respectively.

Open your terminal in the same directory as ``my_app.py``.
We can view the configurable aspects of our application using the ``--help`` command; run the following:

.. code-block:: console
   :caption: Checking the configurable components of our app. (We will add configuration groups in a later lesson.)

   $ python my_app.py --help
   my_app is powered by Hydra.
   
   == Configuration groups ==
   Compose your configuration from those groups (group=option)
   
   
   
   == Config ==
   Override anything in the config (foo.bar=value)
   
   player1: ???
   player2: ???


See that our app requires that we configure two fields: ``player1`` and ``player2``.
Let's configure these fields with the string values ``"mario"`` and ``"luigi"``, respectively.
In your console execute the following command:

.. code-block:: console
   :caption: Launching our application from the command line

   $ python my_app.py player1=mario player2=luigi

.. tip::
   You can `add tab-completion <https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/>`_ to your application's command line interface. This is helpful 
   once you start writing applications that have many configurable components.

To inspect the log written by our application, open a Python terminal in the same 
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
   WindowsPath('C:/outputs/2021-10-21/12-58-13')

Print the contents of ``player_log.txt`` and verify that it matches with how we ran our
program:

.. code-block:: pycon
   
   >>> print_file(latest_job / "player_log.txt")
   Game session log:
   Player 1: mario
   Player 2: luigi


Voilà! As demonstrated, our simple application can now be configured and launched from the 
command line. It should be noted that we can still launch our app from a Python 
console, using :func:`~hydra_zen.launch`, as we did :ref:`in the previous tutorial <launch-basic-app>`.

Reference Documentation
=======================
Want a deeper understanding of how hydra-zen and Hydra work?
The following reference materials are especially relevant to this
tutorial section.

- `~hydra_zen.make_config`
- :hydra:`Hydra's Config Store API <tutorials/structured_config/config_store>`
- :hydra:`Hydra's command line override syntax <advanced/override_grammar/basic>`


.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our application.
