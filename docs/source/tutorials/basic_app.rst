========================================
Create and Launch a Basic App with Hydra
========================================

In this tutorial we will create a basic app that we can launch using Hydra.
While the app itself will be trivial, we will see that it is designed to 
be configurable and that it can be run in a reproducible way.

.. admonition:: Prerequisites

   This tutorial does not assume that you have any familiarity with
   hydra-zen or Hydra. It does, however, assume that you are comfortable
   with using Python. 
   
   If you are not comfortable with using Python, consider consulting this
   resource for `getting started with Python <https://www.pythonlikeyoumeanit.com/module_1.html>`_, and this `tutorial on the essentials of Python <https://www.pythonlikeyoumeanit.com/module_2.html>`_.


Getting Started
===============

We will install hydra-zen and then we will create a Python script where we will create 
our app.

Installing hydra-zen
--------------------

To install hydra-zen, run the following command in your terminal:

.. code:: shell
    
    pip install hydra-zen


To verify that hydra-zen is installed as-expected, open a Python console and try 
importing ``hydra_zen``.


.. code:: pycon
    
    >>> import hydra_zen


Creating a Script for our App
-----------------------------

Navigate to (or create) a directory where you are comfortable with files being written; 
running our app will leave behind some "artifacts" here. Create a new text file called
``my_app.py`` and open it in an editor.

Creating a Simple App
=====================

Let's design this app to take in two (configurable) player names, and to log the 
players' names to a text file called ``player_log.txt``.

Our app will consist of two components:

1. A "config", which defines the configurable interface of our app.
2. A task function, which accepts the populated config, and whose body determines the task that will be completed.


Writing the App
---------------

In ``my_app.py`` we'll define a config and task function for this app. We will use
:func:`~hydra_zen.make_config` to create the config. Write the following 
code in this file.


.. code:: python
    
    # contents of my_app.py
    
    from hydra_zen import make_config, instantiate
    
    Config = make_config("player1", "player2")
    
    def task_function(cfg: Config):
        cfg = instantiate(cfg)
        
        # access the player names from the config
        p1 = cfg.player1
        p2 = cfg.player2

        # write the log with the names
        with open("player_log.txt", "w") as f:
            f.write("Game session log:\n")
            f.write(f"Player 1: {p1}\n" f"Player 2: {p2}")


Launching the App
-----------------

It's time to run our app. Open a Python or IPython console - or a Jupyter notebook - in 
the same directory as ``my_app.py``. First, we will import our config and our task function.


.. code:: pycon
    
    >>> from my_app import Config, task_function

We will also need to import hydra-zen's :func:`~hydra_zen.launch` function.

.. code:: pycon
    
    >>> from hydra_zen import launch

Next, we will launch our app by providing the :func:`~hydra_zen.launch` function with: 
our config, our task function, and specific configured values for the player's names.

.. code:: pycon

   >>> job = launch(Config, task_function, overrides=["player1=link", "player2=zelda"])

Let's inspect the completion status of this job by inspecting ``job.status``; it should
indicate ``COMPLETED``.

.. code:: pycon

   >>> job.status
   <JobStatus.COMPLETED: 1>

.. warning::
   If you modify the contents of ``my_app.py``, then you need to restart your Python 
   console (or restart the kernel of your Jupyter notebook) and re-launch the app in 
   order for these changes to take effect.

Inspecting the Results
----------------------

Our app was designed to log the names of the players for that particular game session;
let's check that this log was written as-expected, and familiarize ourselves with the
other files that Hydra writes when it launches an app.

First, we'll create a simple Python function that will make it easy to print files 
in our Python console

.. code:: pycon

   >>> from pathlib import Path 
   >>> def print_file(x: Path):
   ...     with x.open("r") as f: 
   ...         print(f.read())

By default, Hydra will create a directory called ``outputs``, and will store the app's 
outputs in a time-stamped subdirectory of the form 
``outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}``. The particular subdirectory for our job is 
provided by ``job.working_dir``.

.. code:: pycon
   
   >>> job_dir = Path(job.working_dir)
   >>> job_dir  # output will vary based on reader's date/time/OS
   WindowsPath('outputs/2021-10-21/10-36-23')

The contents of this directory consists of: the log-file that our app wrote, a 
``.hydra`` directory that details the configurations of this particular job, and a 
log-file written by Hydra.

.. code:: pycon
   
   >>> sorted(job_dir.glob("*"))
   [WindowsPath('outputs/2021-10-21/10-36-23/.hydra'),
    WindowsPath('outputs/2021-10-21/10-36-23/player_log.txt'),
    WindowsPath('outputs/2021-10-21/10-36-23/zen_launch.log')]

Let's verify that our app wrote the player-log as-expected.

.. code:: pycon
   
   >>> print_file(job_dir / "player_log.txt")
   Game session log:
   Player 1: link
   Player 2: zelda

Great! The player's names were recorded correctly.

The contents of the ``.hydra`` subdirectory is a collection of YAML files:

.. code:: pycon
   
   >>> sorted((job_dir / ".hydra").glob("*"))
   [WindowsPath('outputs/2021-10-21/10-36-23/.hydra/config.yaml'),
    WindowsPath('outputs/2021-10-21/10-36-23/.hydra/hydra.yaml'),
    WindowsPath('outputs/2021-10-21/10-36-23/.hydra/overrides.yaml')]

To see the particular config that was passed to our task function for this job,
we can inspect ``config.yaml``.

.. code:: pycon
   
   >>> print_file(job_dir / ".hydra" / "config.yaml")
   player1: link
   player2: zelda

We successfully designed, configured, and launched an app using hydra-zen and Hydra!
In the next tutorial, we will add a command line interface to this app. We will also
see how we can reproduce the results of a job by using the breadcrumb trail of configs
that Hydra leaves for us.

.. admonition:: References

   Refer to :func:`~hydra_zen.make_config` for more details about designing configs, including creating configs with default  values, and with type-annotations for type-checking.

   Refer to :func:`~hydra_zen.launch` to learn more about the ``JobReturn`` object that
   is produced by our job, and to see an app run in a multirun fashion.

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our app.

