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
    resolve_execution_whitelist,
    zen_targets,
)

requires_target_policy = pytest.mark.skipif(
    not HYDRA_HAS_TARGET_POLICY, reason="requires hydra-core 1.3.6+"
)
requires_execution_policy = pytest.mark.skipif(
    not HYDRA_HAS_EXECUTION_POLICY, reason="requires hydra-core 1.4+"
)


class A:
    def __init__(self, x=1):
        self.x = x


def a_wrapper(f):
    return f


def a_func():
    return 1


def test_builds_and_just_register_their_targets():
    builds(A, zen_wrappers=a_wrapper, x=just(a_func))
    targets = zen_targets()
    for obj in (A, a_wrapper, a_func):
        assert f"{obj.__module__}.{obj.__qualname__}" in targets


def test_string_wrapper_paths_are_registered_but_interpolations_are_not():
    builds(A, zen_wrappers=["tests.test_execution_whitelist.a_wrapper", "${w}"])
    assert "tests.test_execution_whitelist.a_wrapper" in zen_targets()
    assert "${w}" not in zen_targets()


def test_value_conversion_targets_and_resolvers_are_registered():
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


@pytest.mark.parametrize(
    "value",
    [None, "mypkg.f", ("mypkg.*",), ["mypkg.f"]],
)
def test_resolve_passes_through_non_auto(value):
    assert resolve_execution_whitelist(value) is value


@pytest.mark.parametrize("value", [AUTO, [AUTO], ("mypkg.*", AUTO)])
def test_resolve_expands_auto(value):
    resolved = resolve_execution_whitelist(value)
    assert AUTO not in resolved
    assert set(zen_targets()) <= set(resolved)
    if not isinstance(value, str):
        assert {v for v in value if v != AUTO} <= set(resolved)


def test_policy_module_is_not_reachable_via_hydra_zen_attributes():
    # an attribute path to Hydra's private policy state would be reachable from configs
    assert not any(
        isinstance(v, types.ModuleType) and v.__name__.startswith("hydra._internal")
        for v in vars(compat).values()
    )


@requires_target_policy
@pytest.mark.parametrize(
    "cfg",
    [
        # blocklisted target resolved by zen_processing
        builds(os.remove, "not-a-file.txt", zen_meta={"_note": 1}),
        # aliases of blocklisted targets
        {
            "_target_": "hydra_zen.funcs.zen_processing",
            "_zen_target": "logging.os.remove",
            "_args_": ["not-a-file.txt"],
        },
        {
            "_target_": "hydra_zen.funcs.zen_processing",
            "_zen_target": "tests.test_execution_whitelist.A",
            "_zen_wrappers": "logging.os.remove",
        },
        {"_target_": "hydra_zen.funcs.get_obj", "path": "logging.os.remove"},
    ],
)
def test_resolved_paths_are_authorized_by_hydras_policy(cfg):
    # Calling Hydra's instantiate directly uses Hydra's blocklist
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(InstantiationException):
            hydra_instantiate(cfg)


@pytest.mark.skipif(HYDRA_HAS_EXECUTION_POLICY, reason="requires hydra-core < 1.4")
def test_whitelist_requires_hydra_1_4():
    with pytest.raises(TypeError, match="1.4"):
        instantiate(builds(A), _execution_whitelist_=AUTO)


@requires_execution_policy
def test_configs_are_trusted_code_by_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert instantiate({"_target_": "tests.test_execution_whitelist.A"}).x == 1


@requires_execution_policy
@pytest.mark.parametrize("whitelist", [AUTO, [AUTO]])
def test_auto_permits_in_code_targets(whitelist):
    cfg = builds(A, zen_wrappers=a_wrapper, x=just(a_func))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = instantiate(cfg, _execution_whitelist_=whitelist)
    assert out.x is a_func


@requires_execution_policy
@pytest.mark.parametrize(
    "cfg",
    [
        {"_target_": "os.getcwd"},
        {"_target_": "hydra_zen.funcs.zen_processing", "_zen_target": "os.getcwd"},
        {
            "_target_": "hydra_zen.funcs.zen_processing",
            "_zen_target": "tests.test_execution_whitelist.A",
            "_zen_wrappers": "os.getcwd",
        },
        {"_target_": "hydra_zen.funcs.get_obj", "path": "os.getcwd"},
    ],
)
def test_auto_rejects_targets_not_generated_in_code(cfg):
    with pytest.raises(InstantiationException, match="whitelist"):
        instantiate(cfg, _execution_whitelist_=AUTO)


@requires_execution_policy
def test_auto_can_be_extended():
    assert instantiate({"_target_": "os.getcwd"}, _execution_whitelist_=[AUTO, "os.*"])


@requires_execution_policy
def test_active_whitelist_context_is_respected():
    from hydra.utils import execution_whitelist

    with execution_whitelist(["tests.test_execution_whitelist.A"]):
        assert instantiate(builds(A)).x == 1
        with pytest.raises(InstantiationException, match="whitelist"):
            instantiate({"_target_": "os.getcwd"})


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
            overrides=["m._target_=os.getcwd"],
            execution_whitelist=AUTO,
            version_base=None,
        ).return_value


@requires_execution_policy
@pytest.mark.usefixtures("cleandir")
def test_launch_forwards_active_whitelist_context():
    from hydra.utils import execution_whitelist

    with execution_whitelist(["tests.test_execution_whitelist.A"]):
        with pytest.raises(InstantiationException, match="whitelist"):
            launch(
                make_config(m=builds(A)),
                _task,
                overrides=["m._target_=os.getcwd"],
                version_base=None,
            ).return_value
