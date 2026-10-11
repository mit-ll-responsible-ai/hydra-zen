# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""Compatibility patches for Hydra's target policy (Hydra 1.3.6+).

Hydra 1.3.6 introduced a policy that authorizes targets, and the results of
targets, during instantiation. These patches teach that policy about hydra-zen's
own types.
"""

from typing import Any


def apply_partial_with_wrapper_patch() -> None:  # pragma: no cover
    """Permit `partial_with_wrapper` to be returned by targets during instantiation.

    Hydra 1.3.6+ rejects `functools.partial` subclasses returned by targets because
    an override could hide the subclass' invocation behavior. `partial_with_wrapper`'s
    only override is that it applies `wrappers` to `func` at call time. Thus we
    authorize `func` (along with its arguments) and each wrapper using Hydra's own
    checks, and then return the `partial_with_wrapper` as-is.
    """
    from ._compatibility import HYDRA_TARGET_POLICY_MODULE

    # private Hydra internals; accessed via `Any` as they are not typed
    policy: Any = HYDRA_TARGET_POLICY_MODULE
    if policy is None:
        return

    import functools

    from hydra._internal.instantiate import _instantiate2

    from hydra_zen.funcs import partial_with_wrapper

    original = policy._mediate_target_result
    if getattr(original, "__hydra_zen_patched__", False):
        return

    # The signature of this private function varies across Hydra versions, thus
    # we pass through all arguments beyond `result`.
    @functools.wraps(original)
    def _mediate_target_result(result: Any, *args: Any, **kwargs: Any) -> Any:
        if type(result) is not partial_with_wrapper:
            return original(result, *args, **kwargs)

        original(
            functools.partial(result.func, *result.args, **result.keywords),
            *args,
            **kwargs,
        )
        for wrapper in result.wrappers:
            original(wrapper, *args, **kwargs)
        return result

    _mediate_target_result.__hydra_zen_patched__ = True  # type: ignore[attr-defined]

    # `_instantiate2` imports this function by name, so both references are patched
    policy._mediate_target_result = _mediate_target_result
    _instantiate2._mediate_target_result = _mediate_target_result  # type: ignore[attr-defined]
