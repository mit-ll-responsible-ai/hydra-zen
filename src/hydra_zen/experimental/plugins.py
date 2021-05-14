# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import Any

from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.plugins.plugin import Plugin
from hydra.utils import instantiate
from omegaconf import DictConfig


class PluginsZen(Plugins):
    """This removes the check plugins to be in a top-level module.

    hydra-zen is a Pythonic approach and therefore configurations
    don't need to be discoverable since any user can import and build
    a configuration for a plugin from any module and location.
    """

    @staticmethod
    def instance(*args: Any, **kwargs: Any) -> "PluginsZen":
        ret = Singleton.instance(PluginsZen, *args, **kwargs)
        assert isinstance(ret, PluginsZen)
        return ret

    def _instantiate(self, config: DictConfig) -> Plugin:
        plugin = instantiate(config)
        assert isinstance(plugin, Plugin)
        return plugin
