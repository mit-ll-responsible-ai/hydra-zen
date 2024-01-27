# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Any, Tuple

import pytest
from hypothesis import example, given, strategies as st
from omegaconf import DictConfig

from hydra_zen import builds, make_config, to_yaml, zen
from hydra_zen._compatibility import HYDRA_VERSION
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.wrapper import Zen
from tests.custom_strategies import everything_except


@zen
def zen_identity(x: int):
    return x


def function(x: int, y: int, z: int = 2):
    return x * y * z


def function_with_args(x: int, y: int, z: int = 2, *args):
    return x * y * z


def function_with_kwargs(x: int, y: int, z: int = 2, **kwargs):
    return x * y * z


def function_with_args_kwargs(x: int, y: int, z: int = 2, *args, **kwargs):
    return x * y * z


def test_zen_basic_usecase():
    @zen
    def f(x: int, y: str):
        return x * y

    Cfg = make_config(x=builds(int, 2), y="cow", unused="unused")
    assert f(Cfg) == 2 * "cow"


def test_zen_repr():
    assert repr(zen(lambda x, y: None)) == "zen[lambda(x, y)](cfg, /)"
    assert (
        repr(zen(pre_call=lambda x: x)(lambda x, y: None))
        == "zen[lambda(x, y)](cfg, /)"
    )
    assert repr(zen(make_config("x", "y"))) == "zen[Config(x, y)](cfg, /)"


@pytest.mark.parametrize(
    "exclude,expected",
    [
        (None, (1, 0)),
        ("y", (1, 2)),
        ("y,", (1, 2)),
        (["y"], (1, 2)),
        ("x,y", (-2, 2)),
        (["x", "y"], (-2, 2)),
    ],
)
@given(unpack_kw=st.booleans())
def test_zen_excluded_param(exclude, expected, unpack_kw):
    zenf = zen(lambda x=-2, y=2, **kw: (x, y), exclude=exclude, unpack_kwargs=unpack_kw)
    conf = dict(x=1, y=0)
    assert zenf(conf) == expected


@pytest.mark.parametrize(
    "target",
    [
        zen(function),
        zen(function_with_args),
        zen(function_with_kwargs),
        zen(function_with_args_kwargs),
        zen_identity,
        zen(lambda x: x),
    ],
)
def test_repr_doesnt_crash(target):
    assert isinstance(repr(target), str)


@pytest.mark.parametrize("precall", [None, lambda x: x])
def test_zen_wrapper_trick(precall):
    def f(x):
        return x

    # E.g. @zen
    #      def f(...)
    z1 = zen(f, pre_call=precall)  # direct wrap
    assert isinstance(z1, Zen) and z1.func is f and z1.pre_call is precall

    # E.g. @zen(pre_call=...)
    #      def f(...)
    z2 = zen(pre_call=precall)(f)  # config then wrap
    assert isinstance(z2, Zen) and z2.func is f and z1.pre_call is precall


@given(seed=st.sampled_from(range(4)))
def test_zen_pre_call_precedes_instantiation(seed: int):
    @zen(pre_call=zen(lambda seed: random.seed(seed)))
    def f(x: int, y: int):
        return x, y

    actual = f(
        make_config(
            x=builds(int, 2),
            y=builds(random.randint, 0, 10),
            seed=seed,
        )
    )

    random.seed(seed)
    expected = f.func(2, random.randint(0, 10))

    assert actual == expected


@given(y=st.sampled_from(range(10)), as_dict_config=st.booleans())
def test_interpolations_are_resolved(y: int, as_dict_config: bool):
    @dataclass
    class AA:
        x: Any

    @zen(unpack_kwargs=True)
    def f(
        dict_,
        list_,
        builds_,
        make_config_,
        direct,
        list_of_dataclasses,
        nested_dataclasses,
        **kw,
    ):
        return (
            dict_,
            list_,
            builds_,
            make_config_,
            direct,
            list_of_dataclasses,
            nested_dataclasses,
            kw["nested"],
        )

    cfg_maker = make_config if not as_dict_config else lambda **kw: DictConfig(kw)
    B = make_config(b="${y}")
    (
        dict_,
        list_,
        builds_,
        make_config_,
        direct,
        list_of_dataclasses,
        nested_dataclasses,
        kw,
    ) = f(
        cfg_maker(
            dict_={"x": "${y}"},
            list_=["${y}"],
            builds_=builds(dict, a="${y}"),
            make_config_=B,
            direct="${y}",
            list_of_dataclasses=[AA(x=1), AA(x="${y}")],
            nested_dataclasses=AA(x=AA(x="${y}")),
            nested=dict(top=dict(bottom="${...y}")),
            y=y,
        )
    )

    assert dict_ == {"x": y}
    assert list_ == [y]
    assert builds_ == {"a": y}
    assert make_config_ == B(b=y)
    assert direct == y
    assert list_of_dataclasses == [AA(1), AA(y)]
    assert kw == {"top": {"bottom": y}}
    assert isinstance(nested_dataclasses, AA)
    assert isinstance(nested_dataclasses.x, AA)
    assert nested_dataclasses.x.x == y
    assert isinstance(make_config_, B)
    assert all(isinstance(x, AA) for x in list_of_dataclasses)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(x=21),
        DictConfig(dict(x=21)),
        make_config(x=21),
        to_yaml(dict(x=21)),
    ],
)
def test_supported_config_types(cfg):
    @zen
    def f(x):
        return x

    assert f(cfg) == 21


def test_zen_resolves_default_factories():
    Cfg = make_config(x=[1, 2, 3])
    assert zen_identity(Cfg) == [1, 2, 3]
    assert zen_identity(Cfg()) == [1, 2, 3]


def test_no_resolve():
    def not_resolved(x):
        assert x == dict(x=1, y="${x}")

    def is_resolved(x):
        assert x == dict(x=1, y=1)

    Cfg = make_config(x=1, y="${x}")
    out = zen(lambda **kw: kw, unpack_kwargs=True, pre_call=is_resolved)(Cfg)
    assert out == dict(x=1, y=1)

    Cfg2 = make_config(x=1, y="${x}")
    out2 = zen(
        lambda **kw: kw,
        unpack_kwargs=True,
        resolve_pre_call=False,
        pre_call=not_resolved,
    )(Cfg2)
    assert out2 == dict(x=1, y=1)


def test_zen_works_on_partiald_funcs():
    from functools import partial

    def f(x: int, y: str):
        return x, y

    zen_pf = zen(partial(f, x=1))

    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` is missing the following fields: y",
    ):
        zen_pf.validate(make_config())

    zen_pf.validate(make_config(y="a"))
    assert zen_pf(make_config(y="a")) == (1, "a")
    zen_pf.validate(make_config(x=2, y="a"))
    assert zen_pf(make_config(x=2, y="a")) == (2, "a")


@given(x=st.integers(-5, 5), y=st.integers(-5, 5), unpack_kw=st.booleans())
def test_zen_cfg_passthrough(x: int, y: int, unpack_kw: bool):
    @zen(unpack_kwargs=unpack_kw)
    def f(x: int, zen_cfg, **kw):
        return (x, zen_cfg)

    x_out, cfg = f({"x": x, "y": y, "z": "${y}"})
    assert x_out == x
    assert cfg == {"x": x, "y": y, "z": y}


@given(x=st.integers(-5, 5), wrap_mode=st.sampled_from(["decorator", "inline"]))
def test_custom_zen_wrapper(x, wrap_mode):
    class MyZen(Zen):
        CFG_NAME: str = "secret_cfg"

        def __call__(self, cfg) -> Tuple[Any, str]:
            return (super().__call__(cfg), "moo")

    if wrap_mode == "decorator":

        @zen(ZenWrapper=MyZen)
        def f(x: int, secret_cfg):
            return x, secret_cfg

    elif wrap_mode == "inline":

        def _f(x: int, secret_cfg):
            return x, secret_cfg

        f = zen(_f, ZenWrapper=MyZen)
    else:
        assert False

    cfg = {"x": x}
    f.validate(cfg)

    (out_x1, cfg), moo = f(cfg)  # type: ignore
    assert out_x1 == x
    assert cfg == {"x": x}
    assert moo == "moo"


class A:
    def f(self, x: int, y: int, z: int = 2):
        return x * y * z


method = A().f


@pytest.mark.parametrize(
    "func",
    [
        function,
        function_with_args,
        function_with_kwargs,
        function_with_args_kwargs,
        method,
    ],
)
@pytest.mark.parametrize(
    "cfg",
    [
        make_config(),  # missing x & y
        make_config(not_a_field=2),
        make_config(x=1),  # missing y
        make_config(y=2, z=4),  # missing x
    ],
)
def test_zen_validation_cfg_missing_parameter(cfg, func):
    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` is missing the following fields",
    ):
        zen(func).validate(cfg)


@example(to_yaml(["a", "b"]))
@example(["a", "b"])
@given(bad_config=everything_except((dict, str)))
def test_zen_validate_bad_config(bad_config):
    @zen
    def f(*a, **k): ...

    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` must be a ",
    ):
        f(bad_config)


def test_validate_unpack_kwargs():
    with pytest.raises(TypeError, match=r"`unpack_kwargs` must be type `bool`"):
        zen(lambda a: None, unpack_kwargs="apple")  # type: ignore


def test_zen_validation_excluded_param():
    zen(lambda x: ..., exclude="x").validate(make_config())


def test_zen_validation_cfg_has_bad_pos_args():
    def f(x):
        return x

    @dataclass
    class BadCfg:
        _args_: int = 1  # bad _args_

    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg._args_` must be a sequence type",
    ):
        zen(f).validate(BadCfg)


@pytest.mark.parametrize("not_inspectable", [range(1), False])
def test_zen_validate_no_sig(not_inspectable):
    with pytest.raises(
        HydraZenValidationError,
        match="hydra_zen.zen can only wrap callables that possess inspectable signatures",
    ):
        zen(not_inspectable)


@pytest.mark.parametrize(
    "bad_pre_call", [lambda: None, lambda x, y: None, lambda x, y, z=1: None]
)
def test_pre_call_validates_wrong_num_args(bad_pre_call):
    with pytest.raises(
        HydraZenValidationError,
        match=r"must be able to accept a single positional argument",
    ):
        zen(
            lambda x: None,
            pre_call=bad_pre_call,
        )


def test_pre_call_validates_bad_param_name():
    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` is missing the following fields",
    ):
        zen(
            lambda x: None,
            pre_call=zen(lambda missing: None),
        ).validate({"x": 1})


@pytest.mark.parametrize(
    "func",
    [
        function,
        function_with_args,
        function_with_kwargs,
        function_with_args_kwargs,
        method,
    ],
)
@given(
    x=st.integers(-10, 10),
    y=st.integers(-10, 10),
    # kwargs=st.dictionaries(st.sampled_from(["z", "not_a_field"]), st.integers()),
    instantiate_cfg=st.booleans(),
    unpack_kw=st.booleans(),
)
def test_zen_call(x: int, y: int, instantiate_cfg, func, unpack_kw):
    cfg = make_config(x=x, y=y)
    if instantiate_cfg:
        cfg = cfg()

    # kwargs.pop("not_a_field", None)
    expected = func(x, y)
    actual = zen(func, unpack_kwargs=unpack_kw)(cfg)
    assert expected == actual


@given(x=st.sampled_from(range(10)), unpack_kw=st.booleans())
def test_zen_function_respects_with_defaults(x, unpack_kw: bool):
    @zen(unpack_kwargs=unpack_kw)
    def f(x: int = 2, **kw):
        return x

    assert f(make_config()) == 2  # defer to x's default
    assert f(make_config(x=x)) == x  # overwrite x


def raises():
    raise AssertionError("shouldn't have been called!")


@pytest.mark.parametrize(
    "call",
    [
        lambda x: zen_identity(make_config(x=builds(int, x))),
        lambda x: zen_identity(make_config(x=builds(int, x), y=builds(raises))),
        lambda x: zen(lambda x: x, unpack_kwargs=True)(
            dict(x=builds(int, x), y=builds(raises))
        ),
        lambda x: zen(lambda **kw: kw["x"], unpack_kwargs=True, exclude="y")(
            dict(x=builds(int, x), y=builds(raises))
        ),
    ],
)
@given(x=st.sampled_from(range(10)))
def test_instantiation_only_occurs_as_needed(call, x):
    assert call(x) == x


def test_zen_works_with_non_builds():
    bigger_cfg = make_config(super_conf=make_config(a=builds(int)))
    out = zen(lambda super_conf: super_conf)(bigger_cfg)
    assert out.a == 0


class Pre:
    record = []


pre_call_strat = st.just(lambda cfg: Pre.record.append(cfg.x)) | st.just(
    lambda cfg, optional=None: Pre.record.append(cfg.x)
)


@given(
    pre_call=(pre_call_strat | st.lists(pre_call_strat)),
)
def test_multiple_pre_calls(pre_call):
    Pre.record.clear()
    cfg = make_config(x=1, y="a")
    g = zen_identity.func
    assert zen(pre_call=pre_call)(g)(cfg) == 1
    assert Pre.record == [1] * (len(pre_call) if isinstance(pre_call, list) else 1)


@pytest.mark.skipif(
    sys.platform.startswith("win") and bool(os.environ.get("CI")),
    reason="Things are weird on GitHub Actions and Windows",
)
@pytest.mark.usefixtures("cleandir")
def test_hydra_main():
    import subprocess
    from pathlib import Path

    from hydra_zen import load_from_yaml

    path = (Path(__file__).parent / "example_app" / "dummy_zen_main.py").absolute()
    assert not (Path.cwd() / "outputs").is_dir()
    subprocess.run(["python", path, "x=1", "y=2"]).check_returncode()
    assert (Path.cwd() / "outputs").is_dir()

    *_, latest_job = sorted((Path.cwd() / "outputs").glob("*/*"))

    assert load_from_yaml(latest_job / ".hydra" / "config.yaml") == {
        "x": 1,
        "y": 2,
        "z": "${y}",
        "seed": 12,
    }


@pytest.mark.xfail(
    HYDRA_VERSION < (1, 3, 0),
    reason="hydra_main(config_path=...) only supports wrapped task functions starting "
    "in Hydra 1.3.0",
)
@pytest.mark.skipif(
    sys.platform.startswith("win") and bool(os.environ.get("CI")),
    reason="Things are weird on GitHub Actions and Windows",
)
@pytest.mark.parametrize(
    "dir_, name",
    [
        ("dir1", "cfg1"),
        ("dir1", "cfg2"),
        ("dir2", "cfg1"),
        ("dir2", "cfg2"),
        (None, None),
    ],
)
@pytest.mark.usefixtures("cleandir")
def test_hydra_main_config_path(dir_, name):
    # regression test for https://github.com/mit-ll-responsible-ai/hydra-zen/issues/381
    import subprocess
    from pathlib import Path

    from hydra_zen import load_from_yaml

    path = (
        Path(__file__).parent / "example_app" / "zen_main_w_config_path.py"
    ).absolute()
    assert not (Path.cwd() / "outputs").is_dir()

    run_in = ["python", path]

    if dir_ is not None:
        run_in.extend([f"--config-name={name}", f"--config-path={dir_}"])
    else:
        dir_, name = "default", "default"
    subprocess.run(run_in).check_returncode()

    assert (Path.cwd() / "outputs").is_dir()

    *_, latest_job = sorted((Path.cwd() / "outputs").glob("*/*"))

    assert load_from_yaml(latest_job / ".hydra" / "config.yaml") == {
        f"{dir_}_{name}": 1
    }


@pytest.mark.parametrize(
    "zen_func",
    [
        zen(lambda x, **kw: {"x": x, **kw}, unpack_kwargs=True),
        zen(unpack_kwargs=True)(lambda x, **kw: {"x": x, **kw}),
    ],
)
@given(
    x=st.sampled_from(range(-3, 3)),
    # non-string keys should be skipped by unpack_kw
    kw=st.dictionaries(
        st.sampled_from("abcdef") | st.sampled_from([1, 2]), st.integers()
    ),
)
def test_unpack_kw_basic_behavior(zen_func, x, kw):
    inp = dict(x=builds(int, x))
    inp.update(kw)
    out = zen_func(inp)
    expected = {"x": x, **{k: v for k, v in kw.items() if isinstance(k, str)}}
    assert out == expected


def test_unpack_kw_non_redundant():
    x, y, kw = zen(lambda x, y=2, **kw: (x, y, kw), unpack_kwargs=True)(
        dict(x=1, z="${x}")
    )
    assert x == 1
    assert y == 2
    assert kw == {"z": 1}  # x should not be in kw


def zen_extracts_factory_from_instance():
    @dataclass
    class A:
        x: int = 1

    Conf = builds(dict, y=A(), zen_convert={"dataclass": False})
    assert not hasattr(Conf, "y")

    def f(y):
        return y.x

    assert zen(f)(Conf) == 1


def pikl(x):
    return x * 2


zpikl = zen(pikl)


def test_pickle_compatible():
    loaded = pickle.loads(pickle.dumps(zpikl))
    assert loaded({"x": 3}) == pikl(3)


async def test_async_compatible():
    async def foo(x: int):
        return x

    assert await zen(foo)(dict(x=builds(int, 22))) == 22
