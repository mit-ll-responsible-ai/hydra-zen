# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import warnings
from dataclasses import fields, is_dataclass
from typing import Any, Callable, List, Mapping, Optional, Type, Union

from hydra import initialize
from hydra._internal.callbacks import Callbacks
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobReturn, run_job
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, RunMode
from omegaconf import DictConfig, ListConfig, OmegaConf

from hydra_zen._compatibility import HYDRA_VERSION
from hydra_zen._hydra_overloads import instantiate
from hydra_zen.typing._implementations import DataClass, InstOrType


class _NotSet:  # pragma: no cover
    pass


def _store_config(
    cfg: Union[DataClass, Type[DataClass], DictConfig, ListConfig, Mapping[Any, Any]],
    config_name: str = "hydra_launch",
) -> str:
    """Stores configuration object in Hydra's ConfigStore.

    Parameters
    ----------
    cfg : Union[DataClass, DictConfig, Mapping]
        A configuration as a dataclass, configuration object, or a dictionary.

    config_name : str (default: hydra_launch)
        The configuration name used to store the configuration.

    Returns
    -------
    config_name : str
        The configuration name used to store the default configuration.

    Notes
    -----
    The input configuration is registered in the Hydra ConfigStore [1]_ using a
    user-provided config name.

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/structured_config/config_store
    """
    cs = ConfigStore().instance()
    cs.store(name=config_name, node=cfg)
    return config_name


def launch(
    config: Union[InstOrType[DataClass], Mapping[str, Any]],
    task_function: Callable[[Any], Any],
    overrides: Optional[List[str]] = None,
    multirun: bool = False,
    version_base: Optional[Union[str, Type[_NotSet]]] = _NotSet,
    to_dictconfig: bool = False,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
) -> Union[JobReturn, Any]:
    r"""Launch a Hydra job using a Python-based interface.

    `launch` is designed to closely match the interface of the standard Hydra CLI.
    For example, launching a Hydra job from the CLI via::

       $ python my_task.py job/group=group_name job.group.param=1

    corresponds to the following usage of `launch`:

       >>> job = launch(config, task_function, overrides=["job/group=group_name", "job.group.param=1"])

    Parameters
    ----------
    config : DataClass | Type[DataClass] | Mapping[str, Any]
        A config that will be passed to ``task_function``.

    task_function : Callable[[DictConfig], Any]
        The function that Hydra will execute. Its input will be ``config``, which
        has been modified via the specified ``overrides``

    overrides : Optional[List[str]]
        If provided, sets/overrides values in ``config``. See [1]_ and [2]_
        for a detailed discussion of the "grammar" supported by ``overrides``.

    multirun : bool (default: False)
        Launch a Hydra multi-run ([3]_).

    version_base : Optional[str], optional (default=_NotSet)
        Available starting with Hydra 1.2.0.
        - If the `version_base parameter` is not specified, Hydra 1.x will use defaults compatible with version 1.1. Also in this case, a warning is issued to indicate an explicit version_base is preferred.
        - If the `version_base parameter` is `None`, then the defaults are chosen for the current minor Hydra version. For example for Hydra 1.2, then would imply `config_path=None` and `hydra.job.chdir=False`.
        - If the `version_base` parameter is an explicit version string like "1.1", then the defaults appropriate to that version are used.

    to_dictconfig: bool (default: False)
        If ``True``, convert a ``dataclasses.dataclass`` to a ``omegaconf.DictConfig``. Note, this
        will remove Hydra's cabability for validation with structured configurations.

    config_name : str (default: "zen_launch")
        Name of the stored configuration in Hydra's ConfigStore API.

    job_name : str (default: "zen_launch")

    with_log_configuration : bool (default: True)
        If ``True``, enables the configuration of the logging subsystem from the loaded config.

    Returns
    -------
    result : JobReturn | Any
        If ``multirun is False``:
            A ``JobReturn`` object storing the results of the Hydra experiment via the following attributes
                - ``cfg``: Reflects ``config``
                - ``overrides``: Reflects ``overrides``
                - ``return_value``: The return value of the task function
                - ``hydra_cfg``: The Hydra configuration object
                - ``working_dir``: The experiment working directory
                - ``task_name``: The task name of the Hydra job
                - ``status``: A ``JobStatus`` enum reporting whether or not the job completed successfully
        Else:
            Return values of all launched jobs (depends on the Sweeper implementation).

    References
    ----------
    .. [1] https://hydra.cc/docs/advanced/override_grammar/basic
    .. [2] https://hydra.cc/docs/configure_hydra/intro
    .. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run

    Examples
    --------

    **Basic usage**

    Let's define and launch a trivial Hydra app.

    >>> from hydra_zen import make_config, launch, to_yaml

    First, we will define a config, which determines the configurable interface to our
    "app". For the purpose of example, we'll design the "interface" of this config to accept
    two configurable parameters: ``a`` and ``b``.

    >>> Conf = make_config("a", "b")

    Our task function accepts the config as an input and uses it to run some generic functionality.
    For simplicity's sake, let's design this task function to: convert the job's config to a
    yaml-formatted string, print it, and then return the string.

    >>> def task_fn(cfg):
    ...     out = to_yaml(cfg)  # task's input config, converted to yaml-string
    ...     print(out)
    ...     return out

    Now, let's use `launch` to run this task function via Hydra, using particular configured
    values (or, "overrides") for ``a`` and ``b``.

    >>> job_out = launch(Conf, task_fn, overrides=["a=1", "b='foo'"])
    a: 1
    b: foo

    Let's inspect ``job_out`` to see the ways that it summarizes the results of this job.

    >>> job_out.return_value  # the value returned by `task_fn`
    'a: 1\nb: foo\n'

    >>> job_out.working_dir  # where the job's outputs, logs, and configs are saved
    'outputs/2021-10-19/15-27-11'

    >>> job_out.cfg  # the particular config used to run our task-function
    {'a': 1, 'b': 'foo'}

    >>> job_out.overrides  # the overrides that we provides
    ['a=1', "b='foo'"]

    >>> job_out.status  # the job's completion status
    <JobStatus.COMPLETED: 1>

    **Launching a multirun job**

    We can launch multiple runs of our task-function, using various configured values.
    Let's launch a multirun that sweeps over three configurations

    >>> (outputs,) = launch(
    ...     Conf,
    ...     task_fn,
    ...     overrides=["a=1,2,3", "b='bar'"],
    ...     multirun=True,
    ... )
    [2021-10-19 17:50:07,334][HYDRA] Launching 3 jobs locally
    [2021-10-19 17:50:07,334][HYDRA] 	#0 : a=1 b='bar'
    a: 1
    b: bar
    [2021-10-19 17:50:07,434][HYDRA] 	#1 : a=2 b='bar'
    a: 2
    b: bar
    [2021-10-19 17:50:07,535][HYDRA] 	#2 : a=3 b='bar'
    a: 3
    b: bar

    ``outputs`` contains three corresponding ``JobReturns`` instances.

    >>> len(outputs)
    3
    >>> [j.cfg for j in outputs]
    [{'a': 1, 'b': 'bar'}, {'a': 2, 'b': 'bar'}, {'a': 3, 'b': 'bar'}]

    Each run's outputs, logs, and configs are saved to separate working directories

    >>> [j.working_dir for j in outputs]
    ['multirun/2021-10-19/17-50-07\\0',
    'multirun/2021-10-19/17-50-07\\1',
    'multirun/2021-10-19/17-50-07\\2']
    """

    # used for check below
    _num_dataclass_fields = 0
    if is_dataclass(config):
        _num_dataclass_fields = len(fields(config))

    # store config in ConfigStore
    if to_dictconfig and is_dataclass(config):
        # convert Dataclass to a DictConfig
        dictconfig = OmegaConf.create(
            OmegaConf.to_container(OmegaConf.structured(config))
        )
        config_name = _store_config(dictconfig, config_name)
    else:
        config_name = _store_config(config, config_name)

    # Initializes Hydra and add the config_path to the config search path
    with initialize(
        config_path=None,
        job_name=job_name,
        **(
            {}
            if (HYDRA_VERSION < (1, 2, 0) or version_base is _NotSet)
            else {"version_base": version_base}
        )
    ):

        # taken from hydra.compose with support for MULTIRUN
        gh = GlobalHydra.instance()
        assert gh.hydra is not None

        # Load configuration
        cfg = gh.hydra.compose_config(
            config_name=config_name,
            overrides=overrides if overrides is not None else [],
            run_mode=RunMode.RUN if not multirun else RunMode.MULTIRUN,
            from_shell=False,
            with_log_configuration=with_log_configuration,
        )

        callbacks = Callbacks(cfg)
        run_start = (
            callbacks.on_run_start if not multirun else callbacks.on_multirun_start
        )
        run_start(config=cfg, config_name=config_name)

        hydra_context = HydraContext(
            config_loader=gh.config_loader(), callbacks=callbacks
        )

        if not multirun:
            job = run_job(
                hydra_context=hydra_context,
                task_function=task_function,
                config=cfg,
                job_dir_key="hydra.run.dir",
                job_subdir_key=None,
                configure_logging=with_log_configuration,
            )
            callbacks.on_run_end(config=cfg, config_name=config_name, job_return=job)

            # access the result to trigger an exception in case the job failed.
            _ = job.return_value
        else:
            # Instantiate sweeper without using Hydra's Plugin discovery (Zen!)
            sweeper = instantiate(cfg.hydra.sweeper)
            assert isinstance(sweeper, Sweeper)
            sweeper.setup(
                config=cfg,
                hydra_context=hydra_context,
                task_function=task_function,
            )

            task_overrides = OmegaConf.to_container(
                cfg.hydra.overrides.task, resolve=False
            )
            assert isinstance(task_overrides, list)
            job = sweeper.sweep(arguments=task_overrides)
            callbacks.on_multirun_end(config=cfg, config_name=config_name)

    if is_dataclass(config):
        _num_dataclass_fields_after = len(fields(config))
        if (
            _num_dataclass_fields_after == 0
            and _num_dataclass_fields_after < _num_dataclass_fields
        ):
            warnings.warn(
                "Your dataclass-based config was mutated by this run. If you just executed with a "
                "`hydra/launcher` that utilizes cloudpickle (e.g., hydra-submitit-launcher), there is a known "
                "issue with dataclasses (see: https://github.com/cloudpipe/cloudpickle/issues/386). You will have "
                "to restart your interactive environment ro run `launch` again. To avoid this issue you can use the "
                "`launch` option: `to_dictconfig=True`."
            )

    return job
