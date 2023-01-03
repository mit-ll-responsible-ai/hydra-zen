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
from a command line interface (CLI), using Hydra.


Modifying Our Project
=====================

Open ``my_app.py`` in your editor. We will make the following modifications to it:

1. User :func:`hydra_zen.store` to generate for our task function and to store it in Hydra's global config store.
2. Add a ``__main__`` clause to our ``my_app.py`` script so that the script runs our task function.
3. Use :func:`hydra_zen.zen` to wrap the task function and to generate the CLI.

Modify your script to match this:

.. code-block:: python
   :caption: Contents of my_app.py:

   from hydra_zen import store, zen
   
   # 1) `hydra_zen.store generates a config for our task function
   #    and stores it locally under the entry-name "my_app"
   @store(name="my_app")
   def task_function(player1, player2):
       # write the log with the names
       with open("player_log.txt", "w") as f:
           f.write("Game session log:\n")
           f.write(f"Player 1: {player1}\n" f"Player 2: {player2}")

       return player1, player2
      
   # 2) Executing `python my_app.py [...]` will run our task function
   if __name__ == "__main__":
       # 3) We need to add the configs from our local store to Hydra's
       #    global config store 
       store.add_to_hydra_store()
       
       # 4) Our zen-wrapped task function is used to generate
       #    the CLI, and to specify which config we want to use
       #    to configure the app by default
       zen(task_function).hydra_main(config_name="my_app", 
                                     version_base="1.1",
                                     config_path=None,
                                     )


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

- `~hydra_zen.store`
- :hydra:`Hydra's Config Store API <tutorials/structured_config/config_store>`
- :hydra:`Hydra's command line override syntax <advanced/override_grammar/basic>`


.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our application.
