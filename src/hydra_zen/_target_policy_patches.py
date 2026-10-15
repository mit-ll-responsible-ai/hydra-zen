# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""Patches for Hydra's target policy (Hydra 1.3.6+).

Hydra 1.3.6 introduced a policy that authorizes targets, and the results of
targets, during instantiation. These patches teach that policy about hydra-zen's
own targets:

- `partial_with_wrapper` may be returned by targets.
- `zen_processing` and `get_obj`, which resolve *other* objects from import paths,
  have those paths authorized just as Hydra authorizes its own resolvers (e.g.
  `hydra.utils.get_object`); otherwise they would circumvent Hydra's policy.
"""

import functools
import sys
from collections.abc import Sequence
from typing import Any, Callable

_PATCHED_FLAG = "__hydra_zen_patched__"

# hydra-zen targets that resolve other objects from import paths
_ZEN_PROCESSING = "hydra_zen.funcs.zen_processing"
_GET_OBJ = "hydra_zen.funcs.get_obj"


def _replace_function(
    policy: Any, name: str, make_patched: Callable[[Any], Any]
) -> None:  # pragma: no cover
    """Replace `policy.<name>`, including in any loaded Hydra module that imported
    it by name (otherwise whether the patch applies would depend on import order)."""
    original = getattr(policy, name)
    if getattr(original, _PATCHED_FLAG, False):
        return

    patched = functools.wraps(original)(make_patched(original))
    setattr(patched, _PATCHED_FLAG, True)

    for mod_name, mod in list(sys.modules.items()):
        if (mod_name == "hydra" or mod_name.startswith("hydra.")) and getattr(
            mod, name, None
        ) is original:
            setattr(mod, name, patched)


def _qualified_name(obj: Any) -> str:  # pragma: no cover
    return f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"


def apply_partial_with_wrapper_patch() -> None:  # pragma: no cover
    """Permit `partial_with_wrapper` to be returned by targets during instantiation.

    Hydra 1.3.6+ rejects `functools.partial` subclasses returned by targets because
    an override could hide the subclass' invocation behavior. `partial_with_wrapper`'s
    only override is that it applies `wrappers` to `func` at call time. Thus we
    authorize `func` (along with its arguments) and each wrapper using Hydra's own
    checks, and then return the `partial_with_wrapper` as-is.
    """
    from ._compatibility import _target_policy_module

    # private Hydra internals; accessed via `Any` as they are not typed
    policy: Any = _target_policy_module()
    if policy is None:
        return

    from hydra_zen.funcs import partial_with_wrapper

    def make_patched(original: Any) -> Any:
        # The signature of this private function varies across Hydra versions, thus
        # we pass through all arguments beyond `result`.
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

        return _mediate_target_result

    _replace_function(policy, "_mediate_target_result", make_patched)


def apply_resolver_authorization_patch() -> None:  # pragma: no cover
    """Authorize the import paths that `zen_processing` and `get_obj` resolve.

    `zen_processing` resolves `_zen_target` and any string-valued `_zen_wrappers`;
    `get_obj` resolves `path`. Each path is authorized exactly as Hydra authorizes
    the path of one of its own resolvers (`hydra.utils.get_object`): the literal
    path is checked, and then so is the identity of the object that it resolves to
    (which catches aliases such as `logging.os.system`).

    This enforces whatever policy is active: Hydra's blocklist, or an execution
    whitelist. Under `UNSAFE_DISABLE_EXECUTION_CHECKS`, nothing is checked.
    """
    from ._compatibility import _target_policy_module

    # private Hydra internals; accessed via `Any` as they are not typed
    policy: Any = _target_policy_module()
    if policy is None:
        return

    from hydra._internal.utils import _locate
    from hydra.utils import get_object

    unsafe = getattr(policy, "UNSAFE_DISABLE_EXECUTION_CHECKS", None)
    authorize_result = getattr(policy, "_authorize_discovery_result", None)
    # added in Hydra 1.3.7; Hydra 1.3.6 has no protected-result check
    reject_protected = getattr(policy, "_reject_protected_result", None)

    def make_patched(original: Any) -> Any:
        # `rest` is `(execution_whitelist,)` on Hydra 1.4 and empty on Hydra 1.3.x
        def _authorize_discovery_path(
            target: Any, args: Any, kwargs: Any, full_key: str, *rest: Any
        ) -> Any:
            name = _qualified_name(target)
            if name not in (_ZEN_PROCESSING, _GET_OBJ) or (
                unsafe is not None and rest and rest[0] is unsafe
            ):
                return original(target, args, kwargs, full_key, *rest)

            if name == _GET_OBJ:
                paths: list[Any] = [kwargs.get("path", args[0] if args else None)]
            else:
                wrappers = kwargs.get("_zen_wrappers", ())
                # Mirrors `zen_processing`'s handling of `_zen_wrappers`. Note that
                # Hydra passes list-values as `ListConfig`, not `list`.
                if isinstance(wrappers, str) or not isinstance(wrappers, Sequence):
                    wrappers = (wrappers,)
                paths = [kwargs.get("_zen_target"), *wrappers]

            for path in paths:
                if not isinstance(path, str):
                    continue
                # authorize the literal path, as Hydra does for `get_object(path)`
                original(get_object, (path,), {}, full_key, *rest)
                try:
                    obj = _locate(path)
                except Exception:
                    # nothing to authorize: the target will fail to resolve it
                    continue
                # authorize what the path actually resolves to
                if authorize_result is not None:
                    # Hydra 1.4: exactly what Hydra applies to `get_object(path)`
                    authorize_result(path, obj, full_key, *rest)
                else:
                    policy._authorize_resolved_target_identity(
                        obj, path, full_key, *rest
                    )
                    if reject_protected is not None:
                        reject_protected(obj, path, full_key, *rest)
            return None

        return _authorize_discovery_path

    _replace_function(policy, "_authorize_discovery_path", make_patched)
