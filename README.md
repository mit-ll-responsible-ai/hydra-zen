# hydra-zen

[![Automated tests status](https://github.com/mitll-SAFERai/hydra-zen/workflows/Tests/badge.svg)](https://github.com/mitll-SAFERai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
[![Test coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/mitll-SAFERai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
![Python version support](https://img.shields.io/badge/python-3.6%20&#8208;%203.9-blue.svg)

[Hydra](https://github.com/facebookresearch/hydra) is a framework for elegantly configuring complex applications.
Hydra-Zen (this project) provides simple tools to make it convenient to use Hydra in more sophisticated, Python-centric workflows, 
such as configuring machine learning experiments.
Configure your project using the power of Hydra and while enjoying the [Zen of Python](https://www.python.org/dev/peps/pep-0020/)!

## Installation
`hydra-zen` is light weight: its dependencies are `hydra` and `typing-extensions`

```shell script
pip install hydra-zen
```

## Basic Usage

Let's configure a simple image classification experiment
Hydra-Zen provides the `builds` function for conveniently and dynamically creating 
 [structured configs](https://hydra.cc/docs/next/tutorials/structured_config/schema/) for objects,
 which will enable us to easily instantiate and use those objects in our experiment. 
```python
from dataclasses import dataclass
from typing import Any

from torch.optim import Adam
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from hydra_zen import builds, instantiate


@dataclass
class ExperimentConfig:
    optimizer: Any = builds(Adam, lr=1e-5, hydra_partial=True)
    model: Any = builds(resnet18, pretrained=True)
    dataset: Any = builds(
        CIFAR10,
        root=".",
        train=True,
        download=True,
        transform=[
            builds(RandomHorizontalFlip, p=0.33),
            builds(RandomCrop, size=28, padding=2, populate_full_signature=True),
        ],
    )
```

Each `builds(<target>, ...)` call creates a dataclass, which serves as a structured config for that `<target>`.
We can use this to configure an object to be either fully or partially instantiated using a combination of
user-provided and default values.