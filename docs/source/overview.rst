Overview of hydra-zen
=====================

hydra-zen provides simple, Hydra-compatible tools that enable Python-centric workflows for designing, configuring, and running large-scale projects, such as machine learning experiments.
These functions can be broken down into two categories:

1. Functions like :func:`~hydra_zen.builds` and :func:`~hydra_zen.just` make it simple to dynamically :doc:`generate and compose structured configurations </structured_configs>` of your code.
2. :func:`~hydra_zen.experimental.hydra_run` and :func:`~hydra_zen.experimental.hydra_multirun` make it possible to :doc:`launch Hydra jobs </experimental>` and access their results from within a Python session instead of from the commandline. This means that Hydra's powerful plugins – `parameter sweepers <https://hydra.cc/docs/next/plugins/ax_sweeper>`_ and various `job-launchers <https://hydra.cc/docs/next/plugins/submitit_launcher>`_ – can be leveraged within a Jupyter notebook or any other Python process.


What is Hydra?
--------------

`Hydra <https://github.com/facebookresearch/hydra>`_ is a framework for elegantly configuring applications, such as a web app, and launching them from the commandline. It provides a specification for writing rich, yaml-serializable configurations for your code, and makes it simple to configure and launch instances of your application from the commandline using various launchers and parameter-sweeper plugins.

+----------------------------+------------------------------------------+-------------------------------------------------+
| Configuration (yaml)       | Application                              | Launch with Overrides                           |
+============================+==========================================+=================================================+
| .. code:: yaml             | .. code:: python                         | .. code:: shell                                 |
|                            |                                          |                                                 |
|    db:                     |    import hydra                          |    $ python my_app.py db.port=8888              |
|     type: postgres         |    from omegaconf import OmegaConf       |    db:                                          |
|     host: localhost        |                                          |      type: postgres                             |
|     port: 5342             |    @hydra.main(config_name="config")     |      host: localhost                            |
|                            |    def my_app(cfg) -> None:              |      port: 8888                                 |
|                            |        print(OmegaConf.to_yaml(cfg))     |                                                 |
|                            |                                          |                                                 |
|                            |    if __name__ == "__main__":            |                                                 |
|                            |        my_app()                          |                                                 |
|                            |                                          |                                                 |
+----------------------------+------------------------------------------+-------------------------------------------------+


Check out Hydra's `great documentation <https://hydra.cc/>`_ that explains its design and purpose; it has detailed tutorials as well. And give it a star `on Github <https://github.com/facebookresearch/hydra>`_ if you haven't already!


What is hydra-zen?
------------------

hydra-zen is fully compatible with Hydra.
It provides tools that:

  1. Make it easy and ergonomic to generate rich structured configurations of your code.
  2. Enable Hydra-based work flows that never leave Python.

While Hydra excels at configuring and launching traditional software applications, hydra-zen is designed with data science and machine learning practitioners in mind: users whose code may have *many* intricately-configurable components, and whose end-goal is not necessarily to launch a configurable app from the command line, but to run experiments/analysis in a way that is repeatable, scalable, and self-documenting.


Libraries like `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ help to eliminate boilerplate
code – the for-loops and other control-flow logic – associated with training and testing a neural network;
Hydra with hydra-zen follows suite and eliminates the code you would write to, configure, orchestrate, organize the results of your various experiments.


A structured config that you produce using, e.g., :func:`~hydra_zen.builds` is

.. code:: python

   def my_func(a_number: int, text: str):
        pass


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
|            y = self.target_fn(x)                          |            progress_bar_refresh_rate=0.1,|
|            return self.dataloader(TensorDataset(x, y))    |        )                                 |
+-----------------------------------------------------------+------------------------------------------+


