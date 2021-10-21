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

We will install hydra-zen and then create a Python script, where we will create our app.

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

Designing a Simple App
======================

Our app will consist of two components:

1. A "config", which defines the configurable interface of our app.
2. A task function, which uses the populated config to execute some task via Python code.

Let's design this app to take in two (configurable) player names, and to log the players' 
names to a text file.

Writing the App
---------------

In ``my_app.py`` we'll define a config and task function for this app. Write the 
following in this file.


.. code:: python
    
    # contents of my_app.py
    
    from hydra_zen import make_config, instantiate
    
    Config = make_config("player1", "player2")
    
    def task_function(cfg: Config):
        cfg = instantiate(cfg)
        p1 = cfg.player1
        p2 = cfg.player2
        
        # logging the player's names
        with open("player_log.txt", "w") as f:
            f.write(f"Player 1: {p1}\n" f"Player 2: {p2}")

meowth that's right!
