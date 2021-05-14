from dataclasses import dataclass

import pytest
from hydra._internal.core_plugins.basic_launcher import BasicLauncherConf
from hydra._internal.core_plugins.basic_sweeper import BasicSweeperConf
from omegaconf import OmegaConf

from hydra_zen.experimental.plugins import PluginsZen


@dataclass
class UnknownPlugin:
    _target_: str = "nope"


def test_plugin__instantiate_import_error():
    plugins = PluginsZen.instance()
    with pytest.raises(ImportError):
        plugins._instantiate(OmegaConf.structured(UnknownPlugin))


@pytest.mark.parametrize("plugin", [BasicSweeperConf, BasicLauncherConf])
def test_plugin__instantiate_hydra_core(plugin):
    plugins = PluginsZen.instance()
    plugins._instantiate(OmegaConf.structured(plugin))
