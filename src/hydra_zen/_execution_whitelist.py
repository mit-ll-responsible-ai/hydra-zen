# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""Support for Hydra's execution whitelist (Hydra 1.4+).

hydra-zen treats configs as trusted code: by default, hydra-zen instantiates with
Hydra's execution checks disabled. Users can opt into a whitelist, where `"auto"`
expands to every target that hydra-zen generated in-code (via `builds`, `just`,
etc.) during this process.
"""

from collections.abc import Sequence
from typing import Any

from ._compatibility import HYDRA_HAS_EXECUTION_POLICY, _target_policy_module

__all__ = ["AUTO", "register_target", "zen_targets", "resolve_execution_whitelist"]

AUTO = "auto"

# insertion-ordered set of import paths of the targets that hydra-zen generated
_ZEN_TARGETS: dict[str, None] = {}


def register_target(path: str) -> None:
    """Records `path` as a target that hydra-zen generated in-code."""
    _ZEN_TARGETS[path] = None


def zen_targets() -> tuple[str, ...]:
    """Returns the import paths of all targets that hydra-zen has generated in-code,
    during this process."""
    return tuple(_ZEN_TARGETS)


def resolve_execution_whitelist(value: Any) -> Any:
    """Expands `"auto"`, whether passed alone or within a sequence of whitelist
    entries, to the targets that hydra-zen generated in-code. All other values are
    passed through unchanged."""
    if isinstance(value, str):
        return zen_targets() if value == AUTO else value
    if isinstance(value, Sequence) and AUTO in value:
        return tuple(dict.fromkeys([*(v for v in value if v != AUTO), *zen_targets()]))
    return value


def default_execution_whitelist() -> Any:  # pragma: no cover
    """The whitelist hydra-zen uses when the user didn't specify one.

    Defers to an active `hydra.utils.execution_whitelist(...)` context (including
    one established by `hydra.main(execution_whitelist=...)`) if there is one, and
    otherwise disables Hydra's execution checks: hydra-zen treats configs as code.
    """
    assert HYDRA_HAS_EXECUTION_POLICY
    policy: Any = _target_policy_module()
    if policy._get_active_execution_whitelist() is not None:
        return None  # Hydra will use the active whitelist
    return policy.UNSAFE_DISABLE_EXECUTION_CHECKS


def active_execution_whitelist() -> Any:  # pragma: no cover
    """Returns the whitelist of the active `hydra.utils.execution_whitelist(...)`
    context, if any."""
    assert HYDRA_HAS_EXECUTION_POLICY
    policy: Any = _target_policy_module()
    return policy._get_active_execution_whitelist()


def check_whitelist_supported(value: Any) -> None:
    """Raises if a whitelist was requested but Hydra doesn't support whitelists."""
    if value is not None and not HYDRA_HAS_EXECUTION_POLICY:
        import hydra

        raise TypeError(
            f"An execution whitelist was specified ({value!r}), but execution "
            f"whitelists require hydra-core 1.4 or later (installed: "
            f"{hydra.__version__})."
        )
