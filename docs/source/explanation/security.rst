.. meta::
   :description: hydra-zen's threat model: configs are trusted code, and how to opt into Hydra's execution whitelist.

.. _security:

##############################
Configs Are Code: Threat Model
##############################

A Hydra config can choose which Python code runs: instantiating a config imports and
calls its ``_target_``. So a config is code, much like a pickle file is, and it
should be treated with the same level of trust.

hydra-zen's configs are generated *from* your Python code. ``builds(f)`` writes a config
that calls ``f`` because you asked it to. Thus, by default, hydra-zen treats configs as
trusted code.

When Should You Worry?
======================

In the everyday workflow (you write the code, you run it, and you supply the
command-line overrides) there is no attacker. Anyone who can type
``model._target_=os.system`` at your command line could just as easily run Python
directly.

Config values become a risk when they come from someone **less trusted than the
program that instantiates them**, for example:

- a YAML config, or saved ``.hydra/config.yaml``, that someone else wrote or could have
  modified, which you then load and instantiate;
- a service (e.g. a job-submission web UI) that forwards users' overrides to a Hydra
  app running with the service's privileges.

Do not instantiate configs from sources you don't trust. Hydra's own checks on
instantiation are, `in Hydra's words <https://hydra.cc/docs/advanced/execution_whitelist/>`_,
defense in depth, not a security boundary: Hydra has many other ways for configs and
command-line overrides to choose what code runs (launchers, plugins, ``hydra`` settings,
and more).

Remote launchers
----------------

Launchers that run jobs elsewhere (e.g. on a cluster) send the job, including its config,
through some channel (a shared filesystem, a queue, an object store). Hydra's launchers
serialize jobs with pickle, so whoever can write to that channel can run arbitrary code,
regardless of any whitelist. Make sure that only you can write to that channel.

Hydra's Execution Whitelist
===========================

hydra-core 1.4 introduced an execution whitelist, which restricts which targets
instantiation may use [1]_. Without a whitelist, Hydra 1.4 warns on every instantiated
target, and Hydra 1.5 will raise an error.

hydra-zen's behavior:

- **By default, hydra-zen treats configs as trusted code.**
  :func:`~hydra_zen.instantiate` (and therefore :func:`~hydra_zen.zen` and
  :func:`~hydra_zen.launch`) instantiates with Hydra's execution checks disabled,
  unless a whitelist is active, e.g. via ``hydra.main(execution_whitelist=...)``, or
  ``with hydra.utils.execution_whitelist(...):``.
- **You can opt into a whitelist** via ``instantiate(..., _execution_whitelist_=...)``,
  ``launch(..., execution_whitelist=...)``, and
  ``zen(...).hydra_main(..., execution_whitelist=...)``.
- **"auto"** expands to every target that hydra-zen generated in-code (via ``builds``,
  ``just``, ``zen_wrappers``, etc.) during this process. It can be combined with other
  entries, e.g. ``["auto", "torch.*"]``.

.. code-block:: python

   from hydra_zen import builds, instantiate, zen

   def task(x: int) -> None: ...

   Conf = builds(dict, a=1)

   instantiate(Conf, _execution_whitelist_="auto")  # OK: `dict` was built in-code
   instantiate({"_target_": "os.getcwd"}, _execution_whitelist_="auto")  # rejected

   zen(task).hydra_main(config_name="main", execution_whitelist=["auto", "torch.*"])

Enforcing a whitelist rejects any target that isn't on it, including legitimate ones: a
``_target_`` overridden at the command line, targets in YAML configs, or the result of a
factory that returns a class or other callable (Hydra also authorizes callable results).
Add these to the whitelist explicitly.

hydra-zen's ``zen_meta``, ``zen_wrappers``, and ``just`` features resolve objects from
import paths. Whatever policy is active (Hydra's blocklist, or a whitelist) is applied to
those paths too, just as Hydra applies it to its own resolvers such as
``hydra.utils.get_object``.

.. [1] https://hydra.cc/docs/advanced/execution_whitelist/
