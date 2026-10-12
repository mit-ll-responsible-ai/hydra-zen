# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""Support for Hydra's execution whitelist (Hydra 1.4+).

hydra-zen treats configs as trusted code: by default, hydra-zen instantiates with
Hydra's execution checks disabled. Users can opt into a whitelist, where `"auto"`
expands to every target that hydra-zen generated in-code (via `builds`, `just`,
etc.) during this process.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

from ._compatibility import HYDRA_HAS_EXECUTION_POLICY, _target_policy_module

__all__ = ["AUTO", "register_target", "zen_targets", "resolve_execution_whitelist"]

AUTO = "auto"

# insertion-ordered set of import paths of the targets that hydra-zen generated
_ZEN_TARGETS: dict[str, None] = {}

# `tuple(_ZEN_TARGETS)`, cached until the next registration
_ZEN_TARGETS_TUPLE: Optional[tuple[str, ...]] = None

# the targets that hydra-zen itself registers upon import; see `mark_builtin_targets`
_BUILTIN_TARGETS: tuple[str, ...] = ()


def register_target(path: str) -> None:
    """Records `path` as a target that hydra-zen generated in-code."""
    global _ZEN_TARGETS_TUPLE
    if path not in _ZEN_TARGETS:
        _ZEN_TARGETS[path] = None
        _ZEN_TARGETS_TUPLE = None


def zen_targets() -> tuple[str, ...]:
    """Returns the import paths of all targets that hydra-zen has generated in-code,
    during this process."""
    global _ZEN_TARGETS_TUPLE
    if _ZEN_TARGETS_TUPLE is None:
        _ZEN_TARGETS_TUPLE = tuple(_ZEN_TARGETS)
    return _ZEN_TARGETS_TUPLE


def mark_builtin_targets() -> None:
    """Records the current registry as hydra-zen's built-in targets (e.g. those
    used for value conversion)."""
    global _BUILTIN_TARGETS
    _BUILTIN_TARGETS = zen_targets()


def reset_targets(targets: Optional[Iterable[str]] = None) -> None:
    """Replaces the registry with `targets` (default: hydra-zen's built-in targets).

    For testing purposes only."""
    global _ZEN_TARGETS_TUPLE
    _ZEN_TARGETS.clear()
    _ZEN_TARGETS.update(dict.fromkeys(_BUILTIN_TARGETS if targets is None else targets))
    _ZEN_TARGETS_TUPLE = None


def resolve_execution_whitelist(value: Any) -> Any:
    """Expands `"auto"`, whether passed alone or as an entry of a collection of
    whitelist entries, to the targets that hydra-zen generated in-code.

    Non-sequence collections of entries (e.g. sets) are converted to tuples; all
    other values are passed through unchanged."""
    if isinstance(value, str):
        return zen_targets() if value == AUTO else value
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        return value  # e.g. `UNSAFE_DISABLE_EXECUTION_CHECKS` or a policy object
    entries: Any = value if isinstance(value, Sequence) else tuple(value)
    if AUTO not in entries:
        return entries
    return tuple(dict.fromkeys([*(v for v in entries if v != AUTO), *zen_targets()]))


def active_execution_whitelist() -> Any:  # pragma: no cover
    """Returns the whitelist of the active `hydra.utils.execution_whitelist(...)`
    context, if any."""
    assert HYDRA_HAS_EXECUTION_POLICY
    policy: Any = _target_policy_module()
    return policy._get_active_execution_whitelist()


def default_execution_whitelist() -> Any:  # pragma: no cover
    """The whitelist hydra-zen uses when the user didn't specify one.

    Defers to an active `hydra.utils.execution_whitelist(...)` context (including
    one established by `hydra.main(execution_whitelist=...)`) if there is one, and
    otherwise disables Hydra's execution checks: hydra-zen treats configs as code.
    """
    if active_execution_whitelist() is not None:
        return None  # Hydra will use the active whitelist
    policy: Any = _target_policy_module()
    return policy.UNSAFE_DISABLE_EXECUTION_CHECKS


def check_whitelist_supported(value: Any) -> None:
    """Raises if a whitelist was requested but Hydra doesn't support whitelists."""
    if value is not None and not HYDRA_HAS_EXECUTION_POLICY:
        import hydra

        raise TypeError(
            f"An execution whitelist was specified ({value!r}), but execution "
            f"whitelists require hydra-core 1.4 or later (installed: "
            f"{hydra.__version__})."
        )
