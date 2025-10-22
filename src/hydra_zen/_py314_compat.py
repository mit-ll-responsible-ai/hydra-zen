# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""Python 3.14 compatibility patches for Hydra.

This module contains patches to fix compatibility issues with Hydra on Python 3.14.
Must be imported before any Hydra imports occur.
"""

import sys


def apply_hydra_argparse_patch() -> None:
    """Apply Python 3.14 compatibility patch for Hydra's argparse usage.

    Python 3.14 added validation in argparse._check_help that checks if '%'
    is in the help string. Hydra's LazyCompletionHelp object doesn't support
    the 'in' operator, causing a TypeError during argument parser creation.

    This patch temporarily disables the validation during Hydra's parser
    creation, then immediately restores it. This is safe because:
    1. The validation is purely for catching developer errors early
    2. It's new in Python 3.14 (didn't exist in earlier versions)
    3. Hydra's help strings work correctly when actually displayed
    4. We restore normal behavior immediately after
    """
    if sys.version_info < (3, 14):
        return

    try:
        import argparse
        from functools import wraps

        from hydra._internal import utils as hydra_utils

        # Store the original functions
        original_get_args_parser = hydra_utils.get_args_parser
        original_check_help = argparse.ArgumentParser._check_help

        @wraps(original_get_args_parser)
        def patched_get_args_parser():  # type: ignore
            """Patched version that disables help validation during parser creation."""
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

    except (ImportError, AttributeError):
        # If Hydra isn't installed or the structure changed, silently skip
        pass
