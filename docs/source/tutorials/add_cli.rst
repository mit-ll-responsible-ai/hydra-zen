.. _cli-app:

=======================================
Add a Command Line Interface to Our App
=======================================

.. tip:: 
   Hover your cursor over any code block in this tutorial, and a clipboard will appear.
   Click it to copy the contents of the code block.

In this tutorial we will update our app so that it can be configured an launched 
from a command line interface.

.. admonition:: Prerequisites

   This tutorial assumes that you have completed the earlier tutorial: :ref:`basic-app`


Modifying Our App
=================

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
   cs.store(name="config", node=Config)
   
   
   # 2) Tell Hydra what config to use for our task-function
   @hydra.main(config_path=None, config_name="config")
   def task_function(cfg: Config):
       cfg = instantiate(cfg)
       p1 = cfg.player1
       p2 = cfg.player2
   
       with open("player_log.txt", "w") as f:
           f.write("Game session log:\n")
           f.write(f"Player 1: {p1}\n" f"Player 2: {p2}")
   
   
   if __name__ == "__main__":
       # 3) Executing `python my_app.py [...]` will run our task function
       task_function()


Launching Our App from the Command Line
=======================================

With the above modifications to ``my_app.py`` complete, we can launch our app from the 
command line. The following will launch a job with ``mario`` and ``luigi`` as the names
for player 1 and player 2, respectively.

Open your terminal in the same directory as ``my_app.py`` and execute the following 
command.

.. code-block:: console
   :caption: Launching our app from the command line

   $ python my_app.py player1=mario player2=luigi

.. tip::
   You can `add tab-completion <https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/>`_ to your app's command line interface. This is helpful once you 
   start writing apps that have many configurable components.

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
   WindowsPath('C:/outputs/2021-10-21/12-58-13')

Let's verify that our app still operates as-expected; ``player_log.txt`` should read
as follows:

.. code-block:: pycon
   
   >>> print_file(latest_job / "player_log.txt")
   Game session log:
   Player 1: mario
   Player 2: luigi


Voilà! As demonstrated, our app can now be configured and launched from the command 
line. It should be noted that we can still launch our app from a Python console, using
:func:`~hydra_zen.launch`, as we did :ref:`in the previous tutorial <launch-basic-app>`.

.. admonition:: References

   - `Hydra's Config Store API <https://hydra.cc/docs/next/tutorials/structured_config/config_store>`_
   - `Hydra's command line override syntax <https://hydra.cc/docs/next/advanced/override_grammar/basic/>`_


.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our app.
