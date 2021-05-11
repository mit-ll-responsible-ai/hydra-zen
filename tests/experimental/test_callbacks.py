import pytest

from omegaconf import DictConfig
from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn
from hydra.core.config_store import ConfigStore

from hydra_zen import instantiate, builds
from hydra_zen.experimental import hydra_run, hydra_multirun


class CustomCallback(Callback):
    def __init__(self, callback_name):
        self.name = callback_name

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        ...

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs) -> None:
        ...

    def on_run_start(self, config: DictConfig, **kwargs) -> None:
        ...

    def on_run_end(self, config: DictConfig, **kwargs) -> None:
        ...

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        ...

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        ...


cs = ConfigStore.instance()
cs.store(
    group="hydra/callbacks",
    name="test_callback",
    node=dict(test_callback=builds(CustomCallback, callback_name="test")),
)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fn", [hydra_run, hydra_multirun])
def test_hydra_run_with_callback(fn):
    cfg = dict(a=1, b=1)
    job = fn(
        cfg, task_function=instantiate, overrides=["hydra/callbacks=test_callback"]
    )

    if fn == hydra_run:
        assert "test_callback" in job.hydra_cfg.hydra.callbacks

    if fn == hydra_multirun:
        job = job[0][0]
        assert "test_callback" in job.hydra_cfg.hydra.callbacks
