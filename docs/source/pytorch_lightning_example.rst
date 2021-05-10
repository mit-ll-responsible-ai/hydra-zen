Boilerplate-Free ML: An Example Using hydra-zen and PyTorch Lightning
=====================================================================

Let's use Hydra, hydra-zen, and PyTorch Lightning to **configure and train multiple single-layer neural networks without any boilerplate code**.
In this example we will optimize `arbitrary-width universal function approximators <https://en.wikipedia.org/wiki/Universal_approximation_theorem#Arbitrary-width_case>`_  to fit :math:`\cos{x}`
on a restricted domain.
In mathematical notation, we want to solve the following optimization problem:

.. math::

   F(\vec{v}, \vec{w}, \vec{b}; x) &= \sum_{i=1}^{N}{v_{i}\sigma(x w_i + b_i)}

   \vec{v}^*, \vec{w}^*, \vec{b}^* &= \operatorname*{arg\,min}_{\vec{v}, \vec{w}, \vec{b}\in\mathbb{R}^{N}} \;  \|F(\vec{v}, \vec{w}, \vec{b}; x)\ - \cos{x}\|_{2}

   x &\in [-2\pi, 2\pi]

where :math:`N` – the number of "neurons" in our layer – is a hyperparameter.

The following is the boilerplate-free code.

.. code-block:: python

   import math
   from dataclasses import dataclass
   from typing import Any, Callable, Type

   import pytorch_lightning as pl
   import matplotlib.pyplot as plt
   import torch as tr
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim
   from torch.utils.data import DataLoader, TensorDataset

   from hydra_zen import builds, instantiate, just
   from hydra_zen.experimental import hydra_multirun

+-----------------------------------------------------------+------------------------------------------+
| PyTorch Lightning Module                                  | hydra-zen Configuration                  |
+===========================================================+==========================================+
| .. code:: python                                          | .. code:: python                         |
|                                                           |                                          |
|    class UniversalFuncModule(pl.LightningModule):         |    @dataclass                            |
|        """ y = sum(V sigmoid(X W + b))"""                 |    class ExperimentConfig:               |
|                                                           |        optim: Any = builds(              |
|        def __init__(                                      |            optim.Adam,                   |
|            self,                                          |            hydra_partial=True,           |
|            num_neurons: int,                              |            populate_full_signature=True, |
|            optim: Type[optim.Optimizer],                  |        )                                 |
|            dataloader: Type[DataLoader],                  |                                          |
|            target_fn: Callable[[tr.Tensor], tr.Tensor],   |        dataloader: Any = builds(         |
|            training_domain: tr.Tensor,                    |            DataLoader,                   |
|        ):                                                 |            batch_size=25,                |
|            super().__init__()                             |            shuffle=True,                 |
|            self.optim = optim                             |            drop_last=True,               |
|            self.dataloader = dataloader                   |            hydra_partial=True,           |
|            self.training_domain = training_domain         |        )                                 |
|            self.target_fn = target_fn                     |                                          |
|                                                           |        lightning_module: Any = builds(   |
|            self.model = nn.Sequential(                    |            UniversalFuncModule,          |
|                nn.Linear(1, num_neurons),                 |            num_neurons=10,               |
|                nn.Sigmoid(),                              |            optim="${optim}",             |
|                nn.Linear(num_neurons, 1, bias=False),     |            dataloader="${dataloader}",   |
|            )                                              |            target_fn=just(tr.cos),       |
|                                                           |            training_domain=builds(       |
|        def forward(self, x):                              |                tr.linspace,              |
|            return self.model(x)                           |                start=-2 * math.pi,       |
|                                                           |                end=2 * math.pi,          |
|        def configure_optimizers(self):                    |                steps=1000,               |
|            return self.optim(self.parameters())           |                                          |
|                                                           |            ),                            |
|        def training_step(self, batch, batch_idx):         |        )                                 |
|            x, y = batch                                   |                                          |
|            return F.mse_loss(self.model(x), y)            |        trainer: Any = builds(            |
|                                                           |            pl.Trainer,                   |
|        def train_dataloader(self):                        |            max_epochs=100,               |
|            x = self.training_domain.reshape(-1, 1)        |            gpus=0,                       |
|            y = self.target_fn(x)                          |            progress_bar_refresh_rate=0,  |
|            return self.dataloader(TensorDataset(x, y))    |        )                                 |
+-----------------------------------------------------------+------------------------------------------+


.. code-block:: python

   def task(cfg: ExperimentConfig):
       # Hydra recursively instantiates the lightning module, trainer,
       # and all other instantiable aspects of the configuration
       exp = instantiate(cfg)

       # train the model
       exp.trainer.fit(exp.lightning_module)

       # evaluate the model over the domain to assess the fit
       data = exp.lightning_module.training_domain
       final_fit = exp.lightning_module.forward(data.reshape(-1, 1))

       # return the trained model instance and the final fit
       return (
           exp.lightning_module,
           final_fit.detach().numpy().ravel(),
       )

Now we will train our model using different batch-sizes and model-sizes (i.e. number of "neurons" in the layer).


.. code-block:: python

   >>> jobs, = hydra_multirun(
   ...     ExperimentConfig,
   ...     task,
   ...     overrides=[
   ...         "dataloader.batch_size=20, 200",
   ...         "lightning_module.num_neurons=10, 100"
   ...     ],
   ... )
   [2021-05-04 16:19:34,682][HYDRA] Launching 4 jobs locally
   [2021-05-04 16:19:34,683][HYDRA] 	#0 : lightning_module.num_neurons=10 dataloader.batch_size=20
   [2021-05-04 16:19:41,350][HYDRA] 	#1 : lightning_module.num_neurons=10 dataloader.batch_size=200
   [2021-05-04 16:19:43,512][HYDRA] 	#2 : lightning_module.num_neurons=100 dataloader.batch_size=20
   [2021-05-04 16:19:50,319][HYDRA] 	#3 : lightning_module.num_neurons=100 dataloader.batch_size=200

Hydra will `automatically create an output/working directory <https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory>`_ for each job and save an associated yaml configuration file that documents all of the settings that were used to run that job.
The following shows the directories created associated with jobs **0**, **1**, etc.

.. code-block:: shell

   $ tree multirun/2021-05-04/16-19-17
     ├── 0
     │   ├── .hydra
     │   │   ├── config.yaml
     │   │   ├── hydra.yaml
     │   │   └── overrides.yaml
     │   └── lightning_logs/
     ├── 1
     │   ├── .hydra
     │   │   ├── config.yaml
     .   .   .
     .   .   .

Each ``config.yaml`` file can be used to repeat that particular job.

Visualizing our results

.. code-block:: python

   x = instantiate(ExperimentConfig.lightning_module.training_domain)
   target_fn = instantiate(ExperimentConfig.lightning_module.target_fn)

   fig, ax = plt.subplots()
   ax.plot(x, target_fn(x), ls="--", label="True")

   for j in jobs:
       out = j.return_value[1]
       ax.plot(x, out, label=",".join(s.split(".")[-1] for s in j.overrides))

   ax.grid(True)
   ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


.. image:: https://user-images.githubusercontent.com/29104956/117079795-7fc7a280-ad0a-11eb-9916-4fd63cd2e990.png
   :width: 800
   :alt: Alternative text

Voilà! We just configured, trained, saved, and documented multiple neural networks without writing any boilerplate code.
Hydra + hydra-zen + PyTorch Lightning lets us focus on writing the essentials of our scientific software and keep us out of technical debt.


More Examples of hydra-zen in ML Projects
-----------------------------------------
You can check out `this repository <https://github.com/mitll-SAFERai/hydra-zen-examples>`_ for examples of using hydra-zen in larger-scale ML projects.
