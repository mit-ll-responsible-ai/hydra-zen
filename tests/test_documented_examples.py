from hydra_zen import builds, hydrated_dataclass, instantiate


def test_hydrated_simple_example():
    @hydrated_dataclass(target=dict)
    class DictConf:
        x: int = 2
        y: str = "hello"

    assert instantiate(DictConf(x=10)) == dict(x=10, y="hello")


def power(x: float, exponent: float) -> float:
    return x ** exponent


def test_hydrated_with_partial_exampled():
    @hydrated_dataclass(target=power, hydra_partial=True)
    class PowerConf:
        exponent: float = 2.0

    partiald_power = instantiate(PowerConf)
    assert partiald_power(10.0) == 100.0


def test_documented_builds_simple_roundtrip_example():
    assert {"a": 1, "b": "x"} == instantiate(builds(dict, a=1, b="x"))
