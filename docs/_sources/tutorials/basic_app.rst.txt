.. meta::
   :description: A tutorial for creating a basic program using hydra-zen and Hydra.

.. tip:: 
   Hover your cursor over any code block in this tutorial, and a clipboard will appear.
   Click it to copy the contents of the code block.

.. admonition:: Prerequisites

   This tutorial does not assume that you have any familiarity with
   hydra-zen or Hydra. It does, however, assume that you are comfortable
   with using Python. If you are new to Python, consider consulting this resource for 
   `getting started with Python <https://www.pythonlikeyoumeanit.com/module_1.html>`_, 
   and this 
   `tutorial on the essentials of Python <https://www.pythonlikeyoumeanit.com/module_2.html>`_.

.. _basic-app:

================================================
Create and Launch a Basic Application with Hydra
================================================

In this tutorial we will create a basic application that we can configure and launch 
using Hydra, within a Python program (the next section of the tutorial will introduce a 
CLI). Although this project will be trivial, we will be introduced to the 
overarching design that is used for any Hydra-based project, as well to the 
core functionality provided by hydra-zen.


Getting Started
===============

We will install hydra-zen and then we will create a Python script where we will create 
our application.

Installing hydra-zen
--------------------

To install hydra-zen in your Python environment, run the following command in your 
terminal:

.. code:: shell
    
    pip install hydra-zen


To verify that hydra-zen is installed as-expected, open a Python console and try 
importing ``hydra_zen``.


.. code:: pycon
    
    >>> import hydra_zen


Creating a Script for our Application
-------------------------------------

Navigate to (or create) a directory where you are comfortable with files being written; 
running our code will leave behind some "artifacts" here. Create a new text file called
``my_app.py`` and open it in an editor.

Creating a Simple Hydra-Based Application
=========================================

Let's design our program to take in two (configurable) player names, and to log the 
players' names to a text file called ``player_log.txt``.

Our program will consist of two components, which comprises the structure of any 
Hydra-based application:

1. A "config", which defines the configurable interface of our application.
2. A task function, which accepts the populated config, and whose body specifies the code that will be executed when our application is launched.

We will then use `hydra_zen.launch(Config, task_function, overrides=<...>)` to 
configure and run our task-function from within a python program.

Once we create this typical Hydra app, we will iterate on it and simplify things by using how hydra-zen's :func:`hydra_zen.zen` wrapper.

Writing the Application
-----------------------

In ``my_app.py`` we'll define a config and task function for this application. We will 
use :func:`~hydra_zen.make_config` to create the config. Write the following code in 
this file.


.. code-block:: python
    :caption: Contents of my_app.py:
    
    from hydra_zen import make_config, instantiate
    
    Config = make_config("player1", "player2")
    # A dataclass-type with fields
    #  - player1: Any
    #  - player2: Any
    
    def task_function(cfg):
        # cfg: Config
        obj = instantiate(cfg)
        
        # access the player names from the config
        p1 = obj.player1
        p2 = obj.player2

        # write the log with the names
        with open("player_log.txt", "w") as f:
            f.write("Game session log:\n")
            f.write(f"Player 1: {p1}\n" f"Player 2: {p2}")

        return p1, p2 

.. _launch-basic-app:

Launching the Application
-------------------------

It's time to run our application. Open a Python console -- or a Jupyter notebook -- in 
the same directory as ``my_app.py``. First, we will import our config and our task 
function.


.. code-block:: pycon
    
    >>> from my_app import Config, task_function

We will also need to import hydra-zen's :func:`~hydra_zen.launch` function.

.. code-block:: pycon
    
    >>> from hydra_zen import launch

Next, we will launch our application by providing the :func:`~hydra_zen.launch` 
function with: our config, our task function, and specific configured values for the 
player's names. Here, we will use the names ``link`` and ``zelda`` for the names of 
player 1 and player 2, respectively. The ``version_base`` parameter controls some of the default behaviors of Hydra [1]_.


.. code-block:: pycon
   :caption: Launching our application

   >>> job = launch(Config, task_function, overrides=["player1=link", "player2=zelda"], version_base="1.1")


Let's inspect the completion status of this job by inspecting ``job.status``; it should
indicate ``COMPLETED``.

.. code:: pycon

   >>> job.status
   <JobStatus.COMPLETED: 1>

We can also directly access the value that is returned by our task-function.

.. code:: pycon

   >>> job.return_value
   ('link', 'zelda')


.. warning::
   If you modify the contents of ``my_app.py``, then you need to restart your Python 
   console (or restart the kernel of your Jupyter notebook) and re-launch the 
   application in order for these changes to take effect.

Inspecting the Results
----------------------

Our application was designed to log the names of the players for that particular game 
session; let's check that this log was written as-expected, and familiarize ourselves 
with the other files that Hydra writes when it launches an application.

First, we'll create a simple Python function that will make it easy to print files 
in our Python console

.. code-block:: pycon

   >>> from pathlib import Path 
   >>> def print_file(x: Path):
   ...     with x.open("r") as f: 
   ...         print(f.read())

By default, Hydra will create a directory called ``outputs``, and will store the 
application's outputs in a time-stamped subdirectory of the form  
``outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}``. The particular subdirectory for our job is 
provided by ``job.working_dir``.

.. code-block:: pycon
   
   >>> job_dir = Path(job.working_dir)  # type: ignore
   >>> job_dir  # output will vary based on reader's date/time/OS
   WindowsPath('outputs/2021-10-21/10-36-23')

The contents of this directory consists of: the log-file that our application wrote, a 
``.hydra`` directory that details the configurations of this particular job, and a 
log-file written by Hydra.

.. code:: pycon
   
   >>> sorted(job_dir.glob("*"))
   [WindowsPath('outputs/2021-10-21/10-36-23/.hydra'),
    WindowsPath('outputs/2021-10-21/10-36-23/player_log.txt'),
    WindowsPath('outputs/2021-10-21/10-36-23/zen_launch.log')]

Let's verify that our application wrote the player-log as-expected.

.. code:: pycon
   
   >>> print_file(job_dir / "player_log.txt")
   Game session log:
   Player 1: link
   Player 2: zelda

Great! The players' names were recorded correctly.

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

We successfully designed, configured, and launched an application using hydra-zen and 
Hydra! In the next tutorial, we will add a command line interface to this app.

In the final section, let's see how we can simplify some of our code using special features provided by hydra-zen.

Simplifying Things with :func:`hydra_zen.zen`
---------------------------------------------

We can simplify our task function, removing Hydra-specific logic from it, by using :func:`hydra_zen.zen`. 

Let's update our task function in `my_app.py` to be a simple function whose signature determines the fields that will be extracted and instantiated from our config.
Then we can use :func:`hydra_zen.builds`, instead of :func:`hydra_zen.make_config`, to generate our config based off of the task function's signature.


.. code-block:: python
   :caption: Simplified contents of my_app.py:
    
   from hydra_zen import builds
   
   # note: type annotations are not required
   def new_task_function(player1: str, player2: str):
       # write the log with the names
       with open("player_log.txt", "w") as f:
           f.write("Game session log:\n")
           f.write(f"Player 1: {player1}\n" f"Player 2: {player2}")

       return player1, player2

   Config = builds(new_task_function, populate_full_signature=True)
   # `builds` auto-populates the fields of our config based on the 
   # signature of `task_function`
   #
   # `Config` is dataclass-type whose fields are
   #   - player1: str
   #   - player2: str
   #   - _target_: str = "my_app.task_function"


Wrapping this function as

.. code-block:: python
   :caption: Simplified contents of my_app.py:
    
   from hydra_zen import zen

   hydra_compat_task_fn = zen(task_function)

returns a Hydra-compatible task function, which accepts a single input config just like our original task function did. However, the `zen` wrapper adds the logic of extracting
and instantiating the `player1` and `player2` fields from our config for us.

Now we will supply a zen-wrapped version of our task function to :func:`hydra_zen.launch`; all other aspects of launching our app and inspecting its results are unchanged.

.. code-block:: pycon
   :caption: Launching our application

   >>> from my_app import Config, new_task_function
   >>> from hydra_zen import zen, launch
   >>> wrapped_fn = zen(new_task_function)
   >>> job = launch(Config, wrapped_fn, overrides=["player1=link", "player2=zelda"], version_base="1.1")

See that we were able to launch the same app as before, but with some additional benefits:

- Our task function does not have any Hydra-specific logic and can be used for other purposes.
- We can utilize :func:`hydra_zen.builds` to auto-populate our config instead of hand-specifying the fields.

Reference Documentation
=======================
Want a deeper understanding of how hydra-zen and Hydra work?
The following reference materials are especially relevant to this
tutorial section.

- :func:`~hydra_zen.zen`
- :func:`~hydra_zen.builds`
- :func:`~hydra_zen.make_config`
- :func:`~hydra_zen.launch`
- `Hydra's version_base <https://hydra.cc/docs/upgrades/version_base/#internaldocs-banner>`_


.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``outputs`` directory that Hydra created 
   upon launching our application. You can find this in the same directory as your 
   ``my_app.py`` file.

Footnotes
=========
.. [1] Specifically, we want to ensure that Hydra will `change its working directory to its "jobdir" <https://hydra.cc/docs/upgrades/1.1_to_1.2/changes_to_job_working_dir/>`_ when we launch our app.