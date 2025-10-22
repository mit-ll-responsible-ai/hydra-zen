# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# Python 3.14 compatibility: Patch Hydra's LazyCompletionHelp BEFORE any imports
# This must be done first, before hydra.main is imported anywhere
import sys

if sys.version_info >= (3, 14):
    try:
        # Import modules and capture them in a closure so the patched function
        # can reference them later, but they won't pollute the namespace
        def _apply_hydra_314_patch():
            import argparse
            from functools import wraps

            from hydra._internal import utils as hydra_utils

            # Store the original functions
            original_get_args_parser = hydra_utils.get_args_parser
            original_check_help = argparse.ArgumentParser._check_help

            @wraps(original_get_args_parser)
            def patched_get_args_parser():  # type: ignore
                """Patched version that disables help validation during parser creation.

                Python 3.14 added validation in argparse._check_help that checks if '%'
                is in the help string. Hydra's LazyCompletionHelp object doesn't support
                the 'in' operator, causing a TypeError.

                This patch temporarily disables the validation during Hydra's parser
                creation, then immediately restores it. This is safe because:
                1. The validation is purely for catching developer errors early
                2. It's new in Python 3.14 (didn't exist in earlier versions)
                3. Hydra's help strings work correctly when actually displayed
                4. We restore normal behavior immediately after
                """
                # Temporarily disable help validation
                argparse.ArgumentParser._check_help = lambda self, action: None
                try:
                    return original_get_args_parser()
                finally:
                    # Immediately restore normal validation
                    argparse.ArgumentParser._check_help = original_check_help

            # Apply the monkey patch to hydra._internal.utils
            hydra_utils.get_args_parser = patched_get_args_parser

            # Also patch it in sys.modules['hydra.main'] which is the actual module
            # (Note: hydra.main as an attribute is a function, but the module is different)
            if "hydra.main" in sys.modules:
                sys.modules["hydra.main"].get_args_parser = patched_get_args_parser

        _apply_hydra_314_patch()
        # Clean up the helper function from the namespace
        del _apply_hydra_314_patch

    except (ImportError, AttributeError):
        # If Hydra isn't installed or the structure changed, silently skip
        pass

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
