.. hydra-zen documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
Welcome to hydra-zen's documentation!
=====================================

hydra-zen is a Python library that simplifies the process of writing code (be it research-grade or production-grade) that is:

- **Configurable**: you can configure all aspects of your code from the command line (or from a single Python function).
- **Repeatable**: each run of your code will be self-documenting; the full configuration of your software is saved alongside your results.
- **Scalable**: launch multiple runs of your software, be it on your local machine or across multiple nodes on a cluster.

It builds off -- and is fully compatible with -- `Hydra <https://hydra.cc/>`_, a 
framework for elegantly configuring complex applications. hydra-zen helps simplify the 
process of using Hydra by providing convenient functions for creating configs and 
launching Hydra jobs. It also provides novel functionality such :ref:`wrapped instantiation <zen-wrapper>` and :ref:`meta fields <meta-field>` in configs.

.. admonition:: Join the Discussion 💬

   Share ideas, ask questions, and chat with us over at `hydra-zen's discussion board <https://github.com/mit-ll-responsible-ai/hydra-zen/discussions>`_.


.. tip::

   🎓 Using hydra-zen for your research project? `Cite us <https://zenodo.org/record/5584711>`_!

Installation
============

hydra-zen is lightweight: its only dependencies are ``hydra-core`` and 
``typing-extensions``. To install it, run.

.. code:: console

   $ pip install hydra-zen

If instead you want to try out the features in the upcoming version, you can install 
the latest pre-release of hydra-zen with:

.. code:: console

   $ pip install --pre hydra-zen

Learning About hydra-zen
========================

Our docs are divided into four sections: Tutorials, How-Tos, Explanations, and 
Reference.

If you want to get a bird's-eye view of what hydra-zen is all about, or if you are 
completely new to Hydra, check out our **Tutorials**. For folks who are savvy Hydra 
users, our **How-Tos** and **Reference** materials can help acquaint you with the 
unique capabilities that are offered by hydra-zen. Finally, **Explanations** provide 
readers with taxonomies, design principles, recommendations, and other articles that 
will enrich their understanding of hydra-zen and Hydra.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   how_tos
   explanation
   api_reference
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
