# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""A minimal sweeper plugin for exercising ``launch(..., multirun=True)``.

Hydra's plugin registry admits a sweeper only if it is defined under the
``hydra_plugins`` namespace package, so this lives here rather than beside the
tests that use it. ``tests/conftest.py`` puts the enclosing ``plugins``
directory on ``sys.path`` before Hydra builds its registry.
"""

from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper

from hydra_zen import builds


class LocalBasicSweeper(Sweeper):
    def setup(self, *, hydra_context, task_function, config):
        self.hydra_context = hydra_context
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    def sweep(self, arguments):
        assert self.launcher is not None
        assert self.hydra_context is not None

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        override = parser.parse_overrides(arguments)[0]
        key = override.get_key_element()
        sweep = [f"{key}={val}" for val in override.sweep_string_iterator()]
        overrides = [[x] for x in sweep]

        returns = []
        for i, batch in enumerate(overrides):
            result = self.launcher.launch([batch], initial_job_idx=i)[0]
            returns.append(result)

        return [returns]


ConfigStore.instance().store(
    group="hydra/sweeper", name="local_test", node=builds(LocalBasicSweeper)
)
