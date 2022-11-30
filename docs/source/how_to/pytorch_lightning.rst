.. meta::
   :description: hydra-zen can be used to design a boilerplate-free Hydra application for running PyTorch Lightning experiments.

.. _Lightning:

.. admonition:: Prerequisites

   Your must install `PyTorch <https://pytorch.org/>`_ and `PyTorch Lightning <https://
   www.pytorchlightning.ai/>`_ in your Python environment in order to follow this 
   How-To guide.

.. tip::

   Using hydra-zen for your research project? `Cite us <https://zenodo.org/record/5584711>`_! 😊

======================================================================
Run Boilerplate-Free ML Experiments with PyTorch Lightning & hydra-zen
======================================================================

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ is a library designed to 
eliminate the boilerplate code that is associated with training and testing neural 
networks in PyTorch. This is a natural bedfellow of Hydra and hydra-zen, which eliminate the boilerplate associated with designing software that is configurable, repeatable, and scalable.

Let's use Hydra, hydra-zen, and PyTorch Lightning to **configure and train multiple 
single-layer neural networks without any boilerplate code**. For the sake of 
simplicity, we will train it to simply fit :math:`\cos{x}` on 
:math:`x \in [-2\pi, 2\pi]`.

In this "How-To" we will do the following:

1. Define a simple neural network and `lightning module <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_.
2. Create configs for our lighting module, data loader, optimizer, and trainer.
3. Define a task-function for training and testing a model.
4. Train four different models using combinations of two batch-sizes and two model-sizes (i.e. the number of neurons).
5. Analyze our models' results.
6. Load our best model using the checkpoints saved by PyTorch Lightning and the job-config saved by Hydra.

Defining Our Model
==================

Create a script called ``zen_model.py`` (or, open a Jupyter notebook and include the 
following code. Here, we define our single-layer neural network and the `lightning module 
<https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_ that describes how to train and evaluate our model.

.. code-block:: python
   :caption: Contents of ``zen_model.py``

   from typing import Callable, Type
   
   import pytorch_lightning as pl
   import torch as tr
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.optim import Optimizer
   from torch.utils.data import DataLoader, TensorDataset
   
   from hydra_zen.typing import Partial
   
   
   def single_layer_nn(num_neurons: int) -> nn.Module:
       """y = sum(V sigmoid(X W + b))"""
       return nn.Sequential(
           nn.Linear(1, num_neurons),
           nn.Sigmoid(),
           nn.Linear(num_neurons, 1, bias=False),
       )
   
   
   class UniversalFuncModule(pl.LightningModule):
       def __init__(
           self,
           model: nn.Module,
           optim: Partial[Optimizer],
           dataloader: Type[DataLoader],
           target_fn: Callable[[tr.Tensor], tr.Tensor],
           training_domain: tr.Tensor,
       ):
           super().__init__()
           self.optim = optim
           self.dataloader = dataloader
           self.training_domain = training_domain
           self.target_fn = target_fn
   
           self.model = model
   
       def forward(self, x):
           return self.model(x)
   
       def configure_optimizers(self):
           # provide optimizer with model parameters
           return self.optim(self.parameters())
   
       def training_step(self, batch, batch_idx):
           x, y = batch
           # compute |cos(x) - model(x)|^2
           return F.mse_loss(self.model(x), y)
   
       def train_dataloader(self):
           # generate dataset: x, cos(x)
           x = self.training_domain.reshape(-1, 1)
           y = self.target_fn(x)
           return self.dataloader(TensorDataset(x, y))


.. attention::

   :plymi:`Type-annotations <Module5_OddsAndEnds/Writing_Good_Code.html#Type-Hinting>` are **not** required by hydra-zen. However, they do enable :ref:`runtime type-checking of configured values <type-support>` for our app.


Creating Our Configs and Task Function
======================================

Create another script - named ``experiment.py`` - in the same directory as ``zen_model.py``.
Here, we will create the configs for our optimizer, model, data-loader, lightning module,
and trainer. We'll also define the task function that trains and tests our model.


.. code-block:: python
   :caption: Contents of ``experiment.py``

   import math
   
   import torch as tr
   from torch.optim import Adam
   from torch.utils.data import DataLoader
   from zen_model import UniversalFuncModule, single_layer_nn
   
   import pytorch_lightning as pl
   from hydra_zen import builds, make_config, make_custom_builds_fn, instantiate
   
   pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
   
   OptimConf = pbuilds(Adam)
   
   LoaderConf = pbuilds(DataLoader, batch_size=25, shuffle=True, drop_last=True)
   
   ModelConf = builds(single_layer_nn, num_neurons=10)

   # configure our lightning module
   LitConf = pbuilds(
       UniversalFuncModule,
       model=ModelConf,
       target_fn=tr.cos,
       training_domain=builds(
           tr.linspace, start=-2 * math.pi, end=2 * math.pi, steps=1000
       ),
   )
   
   TrainerConf = builds(pl.Trainer, max_epochs=100)
   
   ExperimentConfig = make_config(
       optim=OptimConf,
       dataloader=LoaderConf,
       lit_module=LitConf,
       trainer=TrainerConf,
       seed=1,
   )
   
   
   def task_function(cfg):
       # cfg: ExperimentConfig
       pl.seed_everything(cfg.seed)
   
       obj = instantiate(cfg)
       
       # finish instantiating the lightning module, data-loader, and optimizer
       lit_module = obj.lit_module(dataloader=obj.dataloader, optim=obj.optim)
   
       # train the model
       obj.trainer.fit(lit_module)
   
       # evaluate the model over the domain to assess the fit
       data = lit_module.training_domain
       final_eval = lit_module.forward(data.reshape(-1, 1))
       final_eval = final_eval.detach().cpu().numpy().ravel()
       
       # return the final evaluation of our model:
       # a shape-(N,) numpy-array
       return final_eval

.. admonition:: Be Mindful of What Your Task Function Returns

   We *could* make this task-function return our trained neural network, which would enable
   convenient access to it, in-memory, after our Hydra job completes. However, launching this
   task function in a multirun fashion will train multiple models and thus would keep *all* of
   those models in-memory (and perhaps on-GPU) simultaneously! 
   
   By not returning the model from our task function, we avoid the risk of hitting out-of-memory
   errors when training multiple large models.


Running Our Experiments
========================

We will use :func:`hydra_zen.launch` to run four jobs: training our model with all four combinations of:

- a batch-size of 20 and 200
- a model with 10 and 100 neurons

Open a Python console (or Jupyter notebook) in the same directory as ``experiment.py`` 
and run the following code.

.. code-block:: pycon
   :caption: Launching four jobs

   >>> from hydra_zen import launch
   >>> from experiment import ExperimentConfig, task_function
   >>> (jobs,) = launch(  # type: ignore
   ...     ExperimentConfig,
   ...     task_function,
   ...     overrides=[
   ...         "dataloader.batch_size=20,200",
   ...         "lit_module.model.num_neurons=10,100",
   ...     ],
   ...     multirun=True,
   ... )
   [2021-10-24 21:23:32,556][HYDRA] Launching 4 jobs locally
   [2021-10-24 21:23:32,558][HYDRA] 	#0 : dataloader.batch_size=20 lit_module.model.num_neurons=10
   [2021-10-24 21:23:45,809][HYDRA] 	#1 : dataloader.batch_size=20 lit_module.model.num_neurons=100
   [2021-10-24 21:23:58,656][HYDRA] 	#2 : dataloader.batch_size=200 lit_module.model.num_neurons=10
   [2021-10-24 21:24:01,796][HYDRA] 	#3 : dataloader.batch_size=200 lit_module.model.num_neurons=100

Keep this Python console open; we will be making use of ``jobs`` in order to inspect 
our results.

Inspecting Our Results
=======================

Visualizing Our Results
-----------------------

Let's begin inspecting our results by plotting our four models on :math:`x \in [-2\pi, 2\pi]`, alongside the
target function: :math:`\cos{x}`. Continuing to work in our current Python console (or Jupyter notebook), run
the following code and verify that you see the plot shown below.

.. code-block:: pycon
   :caption: Plotting our models

   >>> from hydra_zen import instantiate
   >>> import matplotlib.pyplot as plt
   
   >>> x = instantiate(ExperimentConfig.lit_module.training_domain)
   >>> target_fn = instantiate(ExperimentConfig.lit_module.target_fn)
   
   >>> fig, ax = plt.subplots()
   >>> ax.plot(x, target_fn(x), ls="--", label="Target")

   >>> for j in jobs:
   ...     out = j.return_value
   ...     ax.plot(x, out, label=",".join(s.split(".")[-1] for s in j.overrides))
   ... 
   >>> ax.grid(True)
   >>> ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
   >>> plt.show()

.. image:: https://user-images.githubusercontent.com/29104956/138622935-3a3a960f-301f-477e-b5ab-7f4c741b1f9e.png
   :width: 800
   :alt: Plot of four trained models vs the target function


Loading the Model of Best-Fit 
-----------------------------

The 100-neuron model trained with a batch-size of 20 best fits our target function. 
Let's load the model weights that were saved by PyTorch Lightning during training.

Continuing our work in the same Python console, let's verify that job-1 corresponds to 
our desired model. Verify that you see the following outputs.

.. code-block:: pycon
   :caption: Job 1 corresponds to the 100-neuron model trained with batch-size 20.
   
   >>> best = jobs[1]
   >>> best.cfg.dataloader.batch_size
   20
   >>> best.cfg.lit_module.model.num_neurons
   100

Next, we'll load the config for this job. Recall that Hydra saves a ``.hydra/config.yaml`` file, which contains the complete configuration of this job -- we can reproduce 
all aspects of it from this YAML. 

.. code-block:: pycon
   :caption: Loading the complete config for this job
   
   >>> from hydra_zen import load_from_yaml, get_target, to_yaml
   >>> from pathlib import Path

   >>> outdir = Path(best.working_dir)
   >>> cfg = load_from_yaml(outdir / ".hydra" / "config.yaml")

It is worth printing our this config to appreciate all of the exhaustive details that 
it captures about this job.

.. code-block:: pycon
   
   >>> print(to_yaml(cfg))  # fully details this job's config
   optim:
     _target_: hydra_zen.funcs.zen_processing
     _zen_target: torch.optim.adam.Adam
     _zen_partial: true
     lr: 0.001
     betas:
     - 0.9
     - 0.999
     eps: 1.0e-08
     weight_decay: 0
     amsgrad: false
   dataloader:
     _target_: hydra_zen.funcs.zen_processing
     _zen_target: torch.utils.data.dataloader.DataLoader
     _zen_partial: true
     batch_size: 20
     shuffle: true
     drop_last: true
   lit_module:
     _target_: hydra_zen.funcs.zen_processing
     _zen_target: zen_model.UniversalFuncModule
     _zen_partial: true
     model:
       _target_: zen_model.single_layer_nn
       num_neurons: 100
     target_fn:
       _target_: hydra_zen.funcs.get_obj
       path: torch.cos
     training_domain:
       _target_: torch.linspace
       start: -6.283185307179586
       end: 6.283185307179586
       steps: 1000
   trainer:
     _target_: pytorch_lightning.trainer.trainer.Trainer
     max_epochs: 100
     progress_bar_refresh_rate: 0
   seed: 1

PyTorch Lightning saved the model's trained weights as a ``.ckpt`` file in this job's 
working directory. Let's load these weights and use them to instantiate our lighting 
module.

.. code-block:: pycon
   :caption: Loading our lighting module with trained weights

   >>> *_, last_ckpt = sorted(outdir.glob("**/*.ckpt"))
   >>> LitModule = get_target(cfg.lit_module)

   >>> loaded = LitModule.load_from_checkpoint(
   ...     last_ckpt,
   ...     model=instantiate(cfg.lit_module.model),
   ...     target_fn=instantiate(cfg.lit_module.target_fn),
   ...     training_domain=instantiate(cfg.lit_module.training_domain),
   ...     optim=instantiate(cfg.optim),
   ...     dataloader=instantiate(cfg.dataloader),
   ... )

Finally, let's double check that this loaded model behaves as-expected. Evaluating it 
at :math:`-\pi/2`, :math:`0`, and :math:`\pi/2` should return, approximately, :math:`0`, :math:`1`, and :math:`0`, respectively.

.. code-block:: pycon
   :caption: Checkout our loaded model's behavior
   
   >>> import torch as tr
   >>> loaded(tr.tensor([-3.1415 / 2, 0.0, 3.1415 / 2]).reshape(-1, 1))
   tensor([[0.0218],
           [0.9526],
           [0.0125]], grad_fn=<MmBackward>



.. admonition:: Math Details

   For the interested reader... In this toy-problem we are optimizing `arbitrary-width universal function approximators    <https://en.wikipedia.org/wiki/Universal_approximation_theorem#Arbitrary-width_case>`_ to fit :math:`\cos{x}`
   on :math:`x \in [-2\pi, 2\pi]`.
   In mathematical notation, we want to solve the following optimization problem:
   
   .. math::
   
      F(\vec{v}, \vec{w}, \vec{b}; x) &= \sum_{i=1}^{N}{v_{i}\sigma(x w_i + b_i)}
   
      \vec{v}^*, \vec{w}^*, \vec{b}^* &= \operatorname*{arg\,min}_{\vec{v}, \vec{w}, \vec   {b}\in\mathbb{R}^{N}} \;  \|F(\vec{v}, \vec{w}, \vec{b}; x)\ - \cos{x}\|_{2}
   
      x &\in [-2\pi, 2\pi]
   
   where :math:`N` – the number of "neurons" in our layer – is a hyperparameter.

.. attention:: **Cleaning Up**:
   To clean up after this tutorial, delete the ``multirun`` directory that Hydra 
   created upon launching our app. You can find this in the same directory as your 
   ``experiment.py`` file.

More Examples of Using hydra-zen in ML Projects
===============================================

You can check out `this repository <https://github.com/mit-ll-responsible-ai/hydra-zen-examples>`_ for examples of larger-scale ML projects using hydra-zen.

