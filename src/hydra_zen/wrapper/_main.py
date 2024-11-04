from __future__ import annotations

import copy
import functools
import sys
from typing import Any, Callable

from hydra._internal.callbacks import Callbacks
from hydra._internal.utils import get_args_parser
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobRuntime, JobStatus, env_override
from hydra.initialize import initialize
from hydra.types import RunMode, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

import hydra_zen as hz

_UNSPECIFIED_: Any = object()


def run_job(
    task_function: TaskFunction,
    config_path: str | None = _UNSPECIFIED_,
    config_name: str | None = None,
    version_base: str | None = _UNSPECIFIED_,
):
    args_parser = get_args_parser()
    args = args_parser.parse_args()

    if args.config_name is not None:
        config_name = args.config_name

    with initialize(config_path=None, version_base="1.3"):
        gh = GlobalHydra.instance()
        assert gh.hydra is not None

        if args.help:
            gh.hydra.app_help(config_name=config_name, args_parser=args_parser, args=args)
            sys.exit(0)
        has_show_cfg = args.cfg is not None
        if args.resolve and (not has_show_cfg and not args.help):
            raise ValueError("The --resolve flag can only be used in conjunction with --cfg or --help")
        if args.hydra_help:
            gh.hydra.hydra_help(config_name=config_name, args_parser=args_parser, args=args)
            sys.exit(0)

        num_commands = args.run + has_show_cfg + args.multirun + args.shell_completion + (args.info is not None)
        if num_commands > 1:
            raise ValueError("Only one of --run, --multirun, --cfg, --info and --shell_completion can be specified")
        if num_commands == 0:
            args.run = True

        overrides = args.overrides

        if args.cfg:
            gh.hydra.show_cfg(
                config_name=config_name,
                overrides=overrides,
                cfg_type=args.cfg,
                package=args.package,
            )
            sys.exit(0)

        # Load configuration
        cfg = gh.hydra.compose_config(
            config_name=config_name,
            overrides=overrides,
            run_mode=RunMode.RUN,
            from_shell=False,
        )

        callbacks = Callbacks(cfg)
        run_start = callbacks.on_run_start
        run_start(config=cfg, config_name=config_name)

        HydraConfig.instance().set_config(cfg)
        ret = JobReturn()
        task_cfg = copy.deepcopy(cfg)
        with read_write(task_cfg):
            with open_dict(task_cfg):
                del task_cfg["hydra"]

        ret.cfg = task_cfg
        hydra_cfg = copy.deepcopy(HydraConfig.instance().cfg)
        assert isinstance(hydra_cfg, DictConfig)
        ret.hydra_cfg = hydra_cfg
        overrides = OmegaConf.to_container(cfg.hydra.overrides.task)
        assert isinstance(overrides, list)
        ret.overrides = overrides

        with env_override(hydra_cfg.hydra.job.env_set):
            callbacks.on_job_start(config=cfg, task_function=task_function)

            try:
                ret.return_value = task_function(task_cfg)
                ret.status = JobStatus.COMPLETED
            except Exception as e:
                ret.return_value = e
                ret.status = JobStatus.FAILED

        ret.task_name = JobRuntime.instance().get("name")
        callbacks.on_job_end(config=cfg, job_return=ret)

        callbacks.on_run_end(config=cfg, config_name=config_name, job_return=ret)

        # access the result to trigger an exception in case the job failed.
        _ = ret.return_value

    return ret


def zen_main(
    config_path: str | None = _UNSPECIFIED_,
    config_name: str | None = None,
    version_base: str | None = _UNSPECIFIED_,
) -> Callable[[Any], Any]:
    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main() -> Any:
            return run_job(task_function, config_path=config_path, config_name=config_name, version_base=version_base)

        return decorated_main

    return main_decorator


if __name__ == "__main__":
    import hydra_zen as hz

    def task(hi=1, foo=2):
        print(hi, foo)

    hz.store(hz.make_config(foo=3), name="config")
    hz.store.add_to_hydra_store()

    zen_main(config_name="config", config_path=None, version_base=None)(hz.zen(task))()
