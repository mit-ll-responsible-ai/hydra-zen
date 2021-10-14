import pytest

from hydra_zen import builds, instantiate
from hydra_zen.errors import HydraZenDeprecationWarning
from hydra_zen.experimental import hydra_multirun, hydra_run


@pytest.mark.usefixtures("cleandir")
def test_hydra_run_is_deprecated():
    cfg = builds(dict, a=1, b=1)
    overrides = ["a=1"]
    with pytest.warns(HydraZenDeprecationWarning):
        hydra_run(cfg, instantiate, overrides)


@pytest.mark.usefixtures("cleandir")
def test_hydra_multirun_is_deprecated():
    cfg = builds(dict, a=1, b=1)
    overrides = ["a=1,2"]
    with pytest.warns(HydraZenDeprecationWarning):
        hydra_multirun(cfg, instantiate, overrides)
