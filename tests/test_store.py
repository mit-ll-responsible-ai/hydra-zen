# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest
from hydra.core.config_store import ConfigStore

from hydra_zen import instantiate, just, store

cs = ConfigStore().instance()


def func(a: int, b: int):
    assert isinstance(a, int)
    assert isinstance(b, int)
    return (a, b)


def instantiate_from_repo(
    name: str,
    group: Optional[str] = None,
    package: Optional[str] = None,
    provider: Optional[str] = None,
    **kw,
):
    """Fetches config-node from repo by name and instantiates it
    with provided kwargs"""
    if group is not None:
        item = cs.repo[group][f"{name}.yaml"]
    else:
        item = cs.repo[f"{name}.yaml"]
    assert item.package == package
    assert item.provider == provider
    return instantiate(item.node, **kw)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, a=1, b=2), id="inline"),
        pytest.param(lambda: store(a=1, b=2)(func), id="decorated"),
        pytest.param(lambda: store()(func, a=1, b=2), id="partiald_inline"),
        pytest.param(lambda: store()(a=1, b=2)(func), id="partiald_decorated"),
        pytest.param(lambda: store(a=-22)(a=1, b=-22)(func, b=2), id="kw_overrides"),
        pytest.param(
            lambda: store(name="BAD")(func),
            id="ensure_can_fail1",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(a=22, b=10)(func),
            id="ensure_can_fail2",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_kw_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    assert instantiate_from_repo("func") == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, name="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(name="dunk")(func), id="decorated"),
        pytest.param(lambda: store(name="O1")(func, name="dunk"), id="partiald_inline"),
        pytest.param(
            lambda: store(name="O1")(name="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(name="O1")(name="O2")(func, name="dunk"), id="kw_overrides"
        ),
        pytest.param(
            lambda: store(name="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_name_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    assert instantiate_from_repo("dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, group="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(group="dunk")(func), id="decorated"),
        pytest.param(
            lambda: store(group="O1")(func, group="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: store(group="O1")(group="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(group="O1")(group="O2")(func, group="dunk"), id="kw_overrides"
        ),
        pytest.param(
            lambda: store(group="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_group_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    assert instantiate_from_repo(name="func", group="dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, package="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(package="dunk")(func), id="decorated"),
        pytest.param(
            lambda: store(package="O1")(func, package="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: store(package="O1")(package="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(package="O1")(package="O2")(func, package="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: store(package="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_package_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    assert instantiate_from_repo(name="func", package="dunk", a=1, b=2) == (1, 2)


def special_fn(x, **kw):
    kw["target"] = just(x)
    return kw


def never_call(*a, **k):
    assert False


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, to_config=special_fn, a=1, b=2), id="inline"),
        pytest.param(
            lambda: store(to_config=special_fn, a=1, b=2)(func),
            id="decorated",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=1, b=2)(func, to_config=special_fn),
            id="partiald_inline",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=1, b=2)(to_config=special_fn)(func),
            id="partiald_decorated",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=22, b=2)(to_config=never_call)(
                func, to_config=special_fn, a=1
            ),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: store(to_config=never_call, a=1, b=2)(func),
            id="ensure_can_fail1",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(func, to_config=never_call, a=1, b=2),
            id="ensure_can_fail2",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(func, a=1, b=2),
            id="ensure_can_fail3",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            lambda: store(func, a=1, b=3, to_config=special_fn),
            id="ensure_can_fail4",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_to_config_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    assert instantiate_from_repo(name="func") == dict(a=1, b=2, target=func)


def override_store(
    name: str,
    node: Any,
    group: Optional[str] = None,
    package: Optional[str] = None,
    provider: Optional[str] = None,
):
    assert False


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, store_fn=override_store), id="inline"),
        pytest.param(
            lambda: store(store_fn=override_store)(func),
            id="decorated",
        ),
        pytest.param(
            lambda: store()(store_fn=override_store)(func),
            id="partiald_decorated",
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_store_fn_overrides(apply_store: Callable[[], Any]):
    with pytest.raises(AssertionError):
        apply_store()


@pytest.mark.parametrize(
    "apply_store",
    [
        pytest.param(lambda: store(func, provider="dunk", a=1, b=2), id="inline"),
        pytest.param(lambda: store(provider="dunk")(func), id="decorated"),
        pytest.param(
            lambda: store(provider="O1")(func, provider="dunk"), id="partiald_inline"
        ),
        pytest.param(
            lambda: store(provider="O1")(provider="dunk")(func), id="partiald_decorated"
        ),
        pytest.param(
            lambda: store(provider="O1")(provider="O2")(func, provider="dunk"),
            id="kw_overrides",
        ),
        pytest.param(
            lambda: store(provider="BAD")(func),
            id="ensure_can_fail",
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.usefixtures("configstore_repo")
def test_provider_overrides(apply_store: Callable[[], Any]):
    out = apply_store()
    assert out is func
    assert instantiate_from_repo(name="func", provider="dunk", a=1, b=2) == (1, 2)


@pytest.mark.parametrize("bad_val", [1, True, ("a",)])
@pytest.mark.parametrize("field_name", ["name", "group", "package"])
def test_store_param_validation(bad_val, field_name: str):
    with pytest.raises(TypeError, match=rf"`{field_name}` must be"):
        store(func, **{field_name: bad_val})


@dataclass
class DC:
    ...


dc = DC()


def validate_get_name():
    with pytest.raises(TypeError, match=r"Cannot infer config store entry name"):
        store(dc)
