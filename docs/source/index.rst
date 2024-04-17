.. hydra-zen documentation master file, created by
   sphinx-quickstart on Fri Apr 23 17:23:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. admonition:: Join the Discussion 💬

   Share ideas, ask questions, and chat with us over at `hydra-zen's discussion board <https://github.com/mit-ll-responsible-ai/hydra-zen/discussions>`_.


.. tip::

   🎓 Using hydra-zen for your research project? `Cite us <https://arxiv.org/abs/2201.05647>`_!


=====================================
Welcome to hydra-zen's documentation!
=====================================

hydra-zen is a Python library that makes the `Hydra framework <https://hydra.cc/>`_  
simpler and more elegant to use. Use hydra-zen to design your project to be:

- **Configurable**: Change deeply-nested parameters and swap out entire pieces of your program, all from the command line. 
- **Repeatable**: each run of your code will be self-documenting; the full configuration of your software is saved alongside your results.
- **Scalable**: launch multiple runs of your software, be it on your local machine or across multiple nodes on a cluster.

**hydra-zen eliminates all hand-written yaml configs from your Hydra project**. It does 
so by providing functions that :ref:`dynamically <create-config>` and 
:ref:`automatically <additional-types>` generate dataclass-based configs for your code.
It also provides a :ref:`custom config-store API <zenstore>` and 
:ref:`task-function wrapper <zenwrap>`, which help to eliminate most of the 
Hydra-specific boilerplate from your project.

hydra-zen is fully compatible with Hydra, and is appropriate for use in both rapid 
prototypes and production-grade code. It is also great for designing your data science 
and machine learning research to be reproducible. hydra-zen provides specialized 
support for using NumPy, Jax, PyTorch, and Lightning (a.k.a PyTorch-Lightning) in your 
Hydra application. 


hydra-zen at a glance
=====================

Suppose you have the following library code.

.. code-block:: python
   :caption: Some library code that we want to be able to configure and run from the CLI.

   # Contents of baby_torch.py
   
   # Note: no Hydra/hydra-zen specific code here!

   def relu(x): ...
   def sigmoid(x): ...
   
   class Model:
       def __init__(self, activation, nlayers, logits = False) -> None:
           self.summary = f"Model:\n-{activation=}\n-{nlayers=}\n-{logits=}"
   
   class DataLoader:
       def __init__(self, batch_size = 10, shuffle_batch = True):
           self.summary = f"DataLoader:\n-{batch_size=}\n-{shuffle_batch=}\n"
   
   def train_fn(model: Model, dataloader: DataLoader, num_epochs: int = -1):
       print(f"Training with {num_epochs=}\n")
       print(model.summary, end="\n\n")
       print(dataloader.summary)


We want to be able to configure and run the `train_fn` from the commandline, while being
able to modify all aspects of its inputs, including parameters nested in `Model` and 
`DataLoader`.

`hydra_zen` makes short work of this: we can create and store custom configurations for
all parts of this library code and generate a CLI that reflects the resulting hierarchical config.

.. code-block:: python
   :caption: Using hydra-zen to create a configurable CLI for running `train_fn`

   # Contents of train.py

   from hydra_zen import just, store

   from baby_torch import DataLoader, Model, relu, sigmoid
   
   # Automatically generate and store configs for `Model`
   model_store = store(group="model")
   model_store(Model, name="generic")
   model_store(Model, nlayers=100, name="big")
   model_store(Model, nlayers=2, name="tiny")
   
   # Configure that relu/sigmoid should "just" be imported,
   # not initialized during run.
   activation_store = store(group="model/activation")
   activation_store(just(relu), name="relu")
   activation_store(just(sigmoid), name="sigmoid")
   
   data_store = store(group="dataloader")
   data_store(DataLoader, name="train")
   data_store(DataLoader, shuffle_batch=False, name="test")
   
   # Configure the top-level function that will be executed from
   # the CLI; provide the default model & dataloader configs to
   # use.
   store(
       train_fn,
       hydra_defaults=[
           "_self_",
           # default config: 
           #    - 'big' model using relu activation 
           #    - train-mode dataloader
           {"model": "big"},
           {"model/activation": "relu"},
           {"dataloader": "train"},
       ],
   )
   
   if __name__ == "__main__":
       from hydra_zen import zen
   
       store.add_to_hydra_store()

       # Generate the CLI For train_fn 
       zen(train_fn).hydra_main(
           config_name="train_fn",
           config_path=None,
           version_base="1.3",
       )
       # Hydra will accept configuration options from
       # the CLI and merge them with the stored configs.
       # 
       # hydra-zen then instantiates these configs
       # -- creating the Model & DataLoader instances --
       # and passes them to train_fn, running the training code.
       #
       # Hydra records the exact, reproducible config
       # for each run, and saves the results in an
       # auto-generated, configurable output dir

Now we can configure and run `train_fn` from the CLI exposed by `train.py`:

.. tab-set::

   .. tab-item:: default
      
      .. code-block:: console
         :caption: Running the default config.
      
         $ python train.py
         Training with num_epochs=-1
         
         Model:
         -activation=<function relu at 0x0000016B9C10F280>
         -nlayers=100
         -logits=False
         
         DataLoader:
         -batch_size=10
         -shuffle_batch=True


   .. tab-item:: set epoch & use sigmoid

      .. code-block:: console
         :caption: Training for 2 epochs using sigmoid activation.
      
         $ python train.py num_epochs=2 model/activation=sigmoid
         Training with num_epochs=2
         
         Model:
         -activation=<function sigmoid at 0x00000185640D4280>
         -nlayers=100
         -logits=False
         
         DataLoader:
         -batch_size=10
         -shuffle_batch=True

   .. tab-item:: change model & dataloader

      .. code-block:: console
         :caption: Using tiny model with logits, and use batch size of 22.
      
         $ python train.py model=tiny model.logits=True dataloader.batch_size=22
         Training with num_epochs=-1
         
         Model:
         -activation=<function relu at 0x0000016B9C10F280>
         -nlayers=2
         -logits=True
         
         DataLoader:
         -batch_size=22
         -shuffle_batch=True

Each run's reproducible configuration will be saved as a yaml file; by default Hydra 
places these in a time-stamped directory.

.. code-block:: console
   :caption: Viewing the serialized yaml file: training for 2 epochs w/ sigmoid.

   $ less outputs/2023-03-11/12-13-14/.hydra/config.yaml
   _target_: baby_torch.train_fn
   model:
     _target_: baby_torch.Model
     activation:
       path: baby_torch.sigmoid
       _target_: hydra_zen.funcs.get_obj
     nlayers: 100
     logits: false
   dataloader:
     _target_: baby_torch.DataLoader
     batch_size: 10
     shuffle_batch: true
   num_epochs: 2

hydra-zen works with arbitrary Python code bases; this example happens to mimic a 
machine learning application but hydra-zen is ultimately application agnostic.

You can read more about hydra-zen's config store and its auto-config capabilities `here <https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.ZenStore.html#hydra_zen.ZenStore>`_.

.. admonition:: Attention, Hydra users:

  If you are already using Hydra, let's cut to the chase: **the most important benefit of using hydra-zen is that it automatically and dynamically generates structured configs for you**.
  

  .. code-block:: python
     :caption: Creating a structured config *without hydra-zen*
     
     from dataclasses import dataclass, field
     
     def foo(bar: int, baz: list[str], qux: float = 1.23):
         ...
     
     @dataclass
     class FooConf:
         _target_: str = "__main__.foo"
         bar: int = 2
         baz: list[str] = field(default_factory=lambda: ["abc"])
         qux: float = 1.23


  .. code-block:: python
     :caption: Creating an equivalent structured config *with hydra-zen*

     from hydra_zen import builds

     def foo(bar: int, baz: list[str], qux: float = 1.23):
         ...

     ZenFooConf = builds(foo, bar=2, baz=["abc"], populate_full_signature=True)
  

  This means that it is much **easier and safer** to write and maintain the configs for your Hydra applications:
  
  - Write all of your configs in Python. No more yaml files!
  - Write less, :ref:`stop repeating yourself <dry>`, and get more out of your configs.
  - Get automatic type-safety via :func:`~hydra_zen.builds`'s signature inspection.
  - :ref:`Validate your configs <builds-validation>` before launching your application.
  - Leverage :ref:`auto-config support <additional-types>` for additional types, like :py:class:`functools.partial`, that are not natively supported by Hydra.

hydra-zen also also provides Hydra users with powerful, novel functionality. With it, we can:

- Add :ref:`enhanced runtime type-checking <runtime-type-checking>` for our Hydra application, via :ref:`pydantic <pydantic-support>`, `beartype <https://github.com/beartype/beartype>`_, and other third-party libraries.
- Design configs specialized behaviors, like :ref:`configs with meta-fields <meta-field>`. 
- Leverage a powerful :ref:`functionality-injection framework <zen-wrapper>` in our Hydra applications.
- Run static type-checkers on our config-creation code to catch incompatibilities with Hydra.


Installation
============

hydra-zen is lightweight: its only dependencies are ``hydra-core`` and 
``typing-extensions``. To install it, run:

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

Note that **each page in our reference documentation features extensive examples and explanations** of how the various components of hydra-zen work. Check it out!

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
