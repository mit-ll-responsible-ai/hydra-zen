# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


# noqa: E402
def _monkeypatch_hydra():
    from hydra._internal.config_loader_impl import ConfigLoaderImpl, OverridesParser
    from hydra._internal.config_repository import (
        CachingConfigRepository,
        ConfigRepository,
    )

    def _new_init(self, delegate):
        self.delegate = delegate
        self.cache = {}

    CachingConfigRepository.__init__ = _new_init

    def _parse_overrides_and_create_caching_repo(self, config_name, overrides):
        parser = OverridesParser.create()
        parsed_overrides = parser.parse_overrides(overrides=overrides)
        caching_repo = CachingConfigRepository(
            ConfigRepository(config_search_path=self.config_search_path)
        )
        self._process_config_searchpath(config_name, parsed_overrides, caching_repo)
        return parsed_overrides, caching_repo

    ConfigLoaderImpl._parse_overrides_and_create_caching_repo = (
        _parse_overrides_and_create_caching_repo
    )
    print("teehee")


_monkeypatch_hydra()
del _monkeypatch_hydra

from typing import TYPE_CHECKING

from ._hydra_overloads import (
    MISSING,
    instantiate,
    load_from_yaml,
    save_as_yaml,
    to_yaml,
)
from ._launch import hydra_list, launch, multirun
from .structured_configs import (
    ZenField,
    builds,
    hydrated_dataclass,
    just,
    make_config,
    make_custom_builds_fn,
    mutable_value,
)
from .structured_configs._implementations import (
    BuildsFn,
    DefaultBuilds,
    get_target,
    kwargs_of,
)
from .structured_configs._type_guards import is_partial_builds, uses_zen_processing
from .wrapper import ZenStore, store, zen

__all__ = [
    "builds",
    "BuildsFn",
    "DefaultBuilds",
    "hydrated_dataclass",
    "just",
    "kwargs_of",
    "mutable_value",
    "get_target",
    "MISSING",
    "instantiate",
    "load_from_yaml",
    "save_as_yaml",
    "to_yaml",
    "make_config",
    "ZenField",
    "make_custom_builds_fn",
    "launch",
    "is_partial_builds",
    "uses_zen_processing",
    "zen",
    "hydra_list",
    "multirun",
    "store",
    "ZenStore",
]

if not TYPE_CHECKING:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str
