import pytest

from hydra_zen import builds, instantiate
from hydra_zen.typing import ZenConvert


def test_no_flat_target():
    out = builds(builds(int), zen_convert=ZenConvert(flat_target=False))
    assert out._target_.startswith("types.Builds")
    with pytest.raises(Exception):
        instantiate(out)


@pytest.mark.parametrize("options", [ZenConvert(flat_target=True), ZenConvert()])
def test_flat_target(options: ZenConvert):
    out = builds(builds(int), zen_convert=options)
    assert instantiate(out) == int()
