# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import os
import types
import warnings

import pytest
from hydra.errors import InstantiationException
from hydra.utils import instantiate as hydra_instantiate

import hydra_zen._compatibility as compat
from hydra_zen import builds, instantiate, just, launch, make_config
from hydra_zen._compatibility import HYDRA_HAS_EXECUTION_POLICY, HYDRA_HAS_TARGET_POLICY
from hydra_zen._execution_whitelist import (
    AUTO,
    register_target,
    reset_targets,
    resolve_execution_whitelist,
    zen_targets,
)

requires_target_policy = pytest.mark.skipif(
    not HYDRA_HAS_TARGET_POLICY, reason="requires hydra-core 1.3.6+"
)
requires_execution_policy = pytest.mark.skipif(
    not HYDRA_HAS_EXECUTION_POLICY, reason="requires hydra-core 1.4+"
)

# matches the errors raised by Hydra's target policy (as opposed to errors raised
# by a target that was actually called)
POLICY_ERROR = r"blocklist|blacklist|whitelist"

THIS_MODULE = "tests.test_execution_whitelist"


@pytest.fixture(autouse=True)
def isolated_targets():
    # The `"auto"` registry is process-global; each test sees only hydra-zen's
    # built-in targets plus whatever the test itself builds.
    saved = zen_targets()
    reset_targets()
    yield
    reset_targets(saved)


class A:
    def __init__(self, x=1):
        self.x = x


def a_wrapper(f):
    return f


def a_func():
    return 1


def never_built():
    return 1


def test_builds_and_just_register_their_targets():
    assert f"{THIS_MODULE}.A" not in zen_targets()
    builds(A, zen_wrappers=a_wrapper, x=just(a_func))
    targets = zen_targets()
    for obj in (A, a_wrapper, a_func):
        assert f"{obj.__module__}.{obj.__qualname__}" in targets


def test_string_wrapper_paths_are_registered_but_interpolations_are_not():
    builds(A, zen_wrappers=[f"{THIS_MODULE}.a_wrapper", "${w}"])
    assert f"{THIS_MODULE}.a_wrapper" in zen_targets()
    assert "${w}" not in zen_targets()


def test_failed_builds_does_not_register_its_target():
    with pytest.raises(ValueError):
        builds(never_built, _target_=1)  # reserved field name
    assert f"{THIS_MODULE}.never_built" not in zen_targets()


def test_value_conversion_targets_and_resolvers_are_builtin():
    targets = zen_targets()
    for path in (
        "builtins.set",
        "builtins.frozenset",
        "builtins.complex",
        "collections.Counter",
        "hydra_zen.funcs.get_obj",
        "hydra_zen.funcs.zen_processing",
    ):
        assert path in targets


def test_zen_targets_is_cached_until_registry_changes():
    before = zen_targets()
    assert zen_targets() is before
    register_target(f"{THIS_MODULE}.A")
    after = zen_targets()
    assert after is not before
    assert after == (*before, f"{THIS_MODULE}.A")


@pytest.mark.parametrize(
    "value",
    [None, "mypkg.f", ("mypkg.*",), ["mypkg.f"]],
)
def test_resolve_passes_through_non_auto(value):
    assert resolve_execution_whitelist(value) is value


@pytest.mark.parametrize(
    "value",
    [AUTO, [AUTO], ("mypkg.*", AUTO), {"mypkg.*", AUTO}, frozenset({AUTO})],
)
def test_resolve_expands_auto(value):
    resolved = resolve_execution_whitelist(value)
    assert AUTO not in resolved
    assert set(zen_targets()) <= set(resolved)
    if not isinstance(value, str):
        assert {v for v in value if v != AUTO} <= set(resolved)


def test_resolve_converts_non_sequence_collections_to_tuples():
    assert resolve_execution_whitelist({"mypkg.f"}) == ("mypkg.f",)


def test_policy_module_is_not_reachable_via_hydra_zen_attributes():
    # an attribute path to Hydra's private policy state would be reachable from configs
    assert not any(
        isinstance(v, types.ModuleType) and v.__name__.startswith("hydra._internal")
        for v in vars(compat).values()
    )


def _zen_processing(**kwargs):
    return {"_target_": "hydra_zen.funcs.zen_processing", **kwargs}


@requires_target_policy
@pytest.mark.parametrize(
    "make_cfg",
    [
        # blocklisted target resolved by zen_processing
        lambda: builds(os.remove, "not-a-file.txt", zen_meta={"_note": 1}),
        # aliases of blocklisted targets
        lambda: _zen_processing(
            _zen_target="logging.os.remove", _args_=["not-a-file.txt"]
        ),
        lambda: _zen_processing(
            _zen_target=f"{THIS_MODULE}.A", _zen_wrappers="logging.os.remove"
        ),
        # list-valued wrappers are passed to the target as a `ListConfig`
        lambda: _zen_processing(
            _zen_target=f"{THIS_MODULE}.A",
            _zen_wrappers=[f"{THIS_MODULE}.a_wrapper", "logging.os.remove"],
        ),
        lambda: {"_target_": "hydra_zen.funcs.get_obj", "path": "logging.os.remove"},
    ],
)
def test_resolved_paths_are_authorized_by_hydras_policy(make_cfg):
    # Calling Hydra's instantiate directly uses Hydra's blocklist
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(InstantiationException, match=POLICY_ERROR):
            hydra_instantiate(make_cfg())


@requires_execution_policy
@pytest.mark.parametrize(
    "path", ["os.getcwd", "logging.os.remove", "os.environ.update", "builtins.len"]
)
def test_get_obj_is_authorized_like_hydras_get_object(path):
    def outcome(target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                hydra_instantiate({"_target_": target, "path": path})
                return "allowed"
            except InstantiationException:
                return "rejected"

    assert outcome("hydra_zen.funcs.get_obj") == outcome("hydra.utils.get_object")


@pytest.mark.skipif(HYDRA_HAS_EXECUTION_POLICY, reason="requires hydra-core < 1.4")
def test_whitelist_requires_hydra_1_4():
    with pytest.raises(TypeError, match="1.4"):
        instantiate(builds(A), _execution_whitelist_=AUTO)


@requires_execution_policy
def test_configs_are_trusted_code_by_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert instantiate({"_target_": f"{THIS_MODULE}.A"}).x == 1


@requires_execution_policy
@pytest.mark.parametrize("whitelist", [AUTO, [AUTO], {AUTO}])
def test_auto_permits_in_code_targets(whitelist):
    cfg = builds(A, zen_wrappers=[a_wrapper, a_wrapper], x=just(a_func))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = instantiate(cfg, _execution_whitelist_=whitelist)
    assert out.x is a_func


@requires_execution_policy
@pytest.mark.parametrize(
    "make_cfg",
    [
        lambda: {"_target_": f"{THIS_MODULE}.never_built"},
        lambda: _zen_processing(_zen_target=f"{THIS_MODULE}.never_built"),
        lambda: _zen_processing(
            _zen_target=f"{THIS_MODULE}.A", _zen_wrappers=f"{THIS_MODULE}.never_built"
        ),
        lambda: _zen_processing(
            _zen_target=f"{THIS_MODULE}.A",
            _zen_wrappers=[f"{THIS_MODULE}.a_wrapper", f"{THIS_MODULE}.never_built"],
        ),
        lambda: {
            "_target_": "hydra_zen.funcs.get_obj",
            "path": f"{THIS_MODULE}.never_built",
        },
    ],
)
def test_auto_rejects_targets_not_generated_in_code(make_cfg):
    builds(A, zen_wrappers=a_wrapper)  # A and a_wrapper are in-code targets
    with pytest.raises(InstantiationException, match="whitelist"):
        instantiate(make_cfg(), _execution_whitelist_=AUTO)


@requires_execution_policy
def test_auto_can_be_extended():
    assert instantiate(
        {"_target_": f"{THIS_MODULE}.never_built"},
        _execution_whitelist_=[AUTO, f"{THIS_MODULE}.*"],
    )


@requires_execution_policy
def test_active_whitelist_context_is_respected():
    from hydra.utils import execution_whitelist

    with execution_whitelist([f"{THIS_MODULE}.A"]):
        assert instantiate(builds(A)).x == 1
        with pytest.raises(InstantiationException, match="whitelist"):
            instantiate({"_target_": f"{THIS_MODULE}.never_built"})


def _task(cfg):
    return instantiate(cfg.m)


@requires_execution_policy
@pytest.mark.usefixtures("cleandir")
def test_launch_with_auto():
    Conf = make_config(m=builds(A))
    assert (
        launch(Conf, _task, execution_whitelist=AUTO, version_base=None).return_value.x
        == 1
    )
    with pytest.raises(InstantiationException, match="whitelist"):
        launch(
            Conf,
            _task,
            overrides=[f"m._target_={THIS_MODULE}.never_built"],
            execution_whitelist=AUTO,
            version_base=None,
        ).return_value


@requires_execution_policy
@pytest.mark.usefixtures("cleandir")
def test_launch_forwards_active_whitelist_context():
    from hydra.utils import execution_whitelist

    with execution_whitelist([f"{THIS_MODULE}.A"]):
        with pytest.raises(InstantiationException, match="whitelist"):
            launch(
                make_config(m=builds(A)),
                _task,
                overrides=[f"m._target_={THIS_MODULE}.never_built"],
                version_base=None,
            ).return_value
