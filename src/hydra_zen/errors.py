# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


class HydraZenException(Exception):
    """Generic parent class for exceptions thrown by hydra-zen."""


class HydraZenDeprecationWarning(HydraZenException, FutureWarning):
    """A deprecation warning issued by hydra-zen.

    Notes
    -----
    This is a subclass of FutureWarning, rather than DeprecationWarning, so
    that the warnings that it emits are not filtered by default.
    """


class HydraZenValidationError(HydraZenException):
    pass


class HydraZenUnsupportedPrimitiveError(HydraZenValidationError, ValueError):
    pass
