.. meta::
   :description: The changelog for hydra-zen, including what's new.

=========
Changelog
=========

This is a record of all past hydra-zen releases and what went into them, in reverse 
chronological order. All previous releases should still be available on pip.

.. _v0.8.0:

------------------
0.8.0 - 2022-XX-XX
------------------

.. note:: This is documentation for an unreleased version of hydra-zen


Support for New Hydra/OmegaConf Features
----------------------------------------
OmegaConf ``v2.2.1`` is added native support for the following types:

- :py:class:`pathlib.Path`

hydra-zen :ref:`already provides support for these <additional-types>`, but this version will defer to OmegaConf's native support when possible. (See :pull:`276`)


Improvements
------------
- The :ref:`automatic type refinement <type-support>` performed by :func:`~hydra_zen.builds` now has enhanced support for ``typing.Annotated``, ``typing.NewType``, and ``typing.TypeVarTuple``. (See :pull:`283`)


Bug Fixes
---------
- :func:`~hydra_zen.builds` would raise a ``TypeError`` if it encountered a target whose signature contained the annotations ``ParamSpecArgs`` or  ``ParamSpecKwargs``. It can now sanitize these annotations properly. (See :pull:`283`)


.. _v0.7.1:

------------------
0.7.1 - 2022-06-22
------------------

Bug Fixes
---------

The validation that hydra-zen performs on ``hydra_defaults`` was overly restrictive. E.g. it would flag ``[{"some_group": None}]`` as invalid, even though null is permitted in `Hydra's default list syntax <https://hydra.cc/docs/advanced/defaults_list/>`_.
This patch fixes this validation and updates the docs & annotations for ``hydra_defaults`` in :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config`.
See :pull:`287` for more details. Thanks to ``@mgrinshpon-doxel`` for the bug report.


.. _v0.7.0:

------------------
0.7.0 - 2022-05-10
------------------

New Features
------------

**Support for defaults lists**

Hydra's `defaults list <https://hydra.cc/docs/advanced/defaults_list/>`_ field can be passed to :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config` via the new ``hydra_defaults`` argument. Basic runtime and static type-checking are performed on this field. See :pull:`264` for more details and examples.


**Improved functionality for types with Specialized hydra-zen support**

:func:`~hydra_zen.just`, :func:`~hydra_zen.to_yaml`, and :func:`~hydra_zen.save_as_yaml` can directly 
operate on values of :ref:`types with specialized support from hydra-zen <additional-types>`; these 
values will automatically be converted to structured configs. 

.. code-block:: pycon

   >>> from functools import partial
   >>> from hydra_zen import to_yaml, just

   >>> def f(x): return x**2
   >>> partiald_f = partial(f, x=2)

   >>> just(partiald_f)  # convert to structured config
   PartialBuilds_f(_target_='__main__.f', _partial_=True, x=2)

   >>> print(to_yaml(partiald_f))  # convert to yaml
   _target_: __main__.f
   _partial_: true
   x: 2

See :pull:`250` and :pull:`259` for more details and examples.

Support for Upcoming Hydra/OmegaConf Features
---------------------------------------------
OmegaConf ``v2.2.0`` is adding native support for the following types:

- :py:class:`bytes`

hydra-zen :ref:`already provides support for these <additional-types>`, but this version will defer to OmegaConf's native support when possible. (See :pull:`262`)

OmegaConf ``v2.2.0`` improves its type-checking, with added support for nested 
containers. Accordingly, hydra-zen's :ref:`automatic type refinement <type-support>` 
will no longer auto-broaden nested container types when ``OmegaConf v2.2.0+`` is 
installed. (See :pull:`261`)


Hydra ``v1.2.0`` is introducing a ``version_base`` parameter that can control default behaviors in ``hydra.run`` and ``hydra.initialize``.
Correspondingly, ``version_base`` is now exposed via `~hydra_zen.launch`. See :pull:`273` for more details.


.. _0p7p0-deprecations:

Deprecations
------------
:pull:`263` deprecates the ``builds_bases`` argument in :func:`~hydra_zen.make_custom_builds`. It will 
be removed in hydra-zen v0.8.0. Users will need to specify ``builds_bases`` on a 
per-config basis via ``builds``.


Bug Fixes
---------
- ``hydra_zen.builds(<Child.class-method>)`` would create a config with the wrong target if ``<class-method>`` was defined on a parent of ``Child``. See :issue:`265`.

Improvements
------------
- Fixed internal protocol of ``partial`` to be compatible with latest type-shed annotations.
- Add missing annotation overloads for :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_custom_builds`
- Substantial source code reorganization
- Improved pyright tests

.. _v0.6.0:

------------------
0.6.0 - 2022-03-09
------------------

This release focuses on improving hydra-zen's type-annotations; it increases the 
degree to which IDEs and static-analysis tools can infer information about common
hydra-zen code patterns.

It should be noted that hydra-zen leverages advanced typing features (e.g. recursive 
types) and that some type-checkers do not support these features yet. hydra-zen's type 
annotations are validated by `pyright <https://github.com/microsoft/pyright>`_. Thus we recommend that users leverage pyright and pyright-based language servers in their 
IDEs (e.g. using Pylance in VSCode) for the best experience.

(A note to VSCode users: make sure to set `Type Checking Mode` to `basic` in your IDE -- it is disabled by default!)

Bug Fixes
---------

``builds(<target>, builds_bases=(...))`` now properly supports the case where a parent config introduces zen-processing features via inheritance. See :pull:`236` for more details.


Improvements
------------
- ``builds(<target>, populate_full_signature=True)`` now carries accurate type information about the target's signature. Thus IDEs can now auto-complete the signature of the resulting structured config. See :pull:`224` for examples and details.
- Type-information is now dispatched by :func:`~hydra_zen.make_custom_builds_fn` for the common use-cases of ``populate_full_signature=True`` and ``zen_partial=True``, respectively. See :pull:`224` for examples and details.
- ``hydra_zen.typing.ZenWrappers`` is now a publicly-available annotation. It reflects valid types for ``builds(..., zen_wrappers=<...>)``.
- hydra-zen now has a pyright-verified `type completeness score <https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#verifying-type-completeness>`_ of 100%. Our CI now requires that this score does not drop below 100%. See :pull:`226` for more details.
- Improved compatibility with mypy (:pull:`243`)
 

Support for Upcoming Hydra Features
-----------------------------------

Hydra 1.1.2 will introduce `support for partial instantiation of targeted configs <https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#partial-instantiation>`_ via the ``_partial_`` field. ``builds(<target>, zen_partial=True)`` will now set the ``_partial_`` field on the structured config
rather than using ``hydra_zen.funcs.zen_processing`` to facilitate partial instantiation.


+---------------------------------------------------+---------------------------------------------------+
| .. code-block:: pycon                             | .. code-block:: pycon                             |
|    :caption: Hydra < 1.1.2                        |    :caption: 1.1.2 <= Hydra                       |
|                                                   |                                                   |
|    >>> Conf = builds(dict, a=1, zen_partial=True) |    >>> Conf = builds(dict, a=1, zen_partial=True) |
|                                                   |                                                   |
|    >>> print(to_yaml(Conf))                       |    >>> print(to_yaml(Conf))                       |
|    _target_: hydra_zen.funcs.zen_processing       |    _target_: builtins.dict                        |
|    _zen_target: builtins.dict                     |    _partial_: true                                |
|    _zen_partial: true                             |    a: 1                                           |
|    a: 1                                           |                                                   |
|                                                   |    >>> instantiate(Conf)                          |
|    >>> instantiate(Conf)                          |    functools.partial(<class 'dict'>, a=1)         |
|    functools.partial(<class 'dict'>, a=1)         |                                                   |
+---------------------------------------------------+---------------------------------------------------+


This change will only occur when one's locally-installed version of ``hydra-core`` is 1.1.2 or higher. Structured configs and yamls that configure partial'd objects via ``hydra_zen.funcs.zen_processing`` are still valid and will instantiate in the same way as before. I.e. this is only a compatibility-breaking change for code that relied on the specific implementation details of the structured config produced by ``builds(<target>, zen_partial=True)``.

In accordance with this change, the definition of ``hydra_zen.typing.PartialBuilds`` has been changed; it now reflects a union of protocols: ``ZenPartialBuilds[T] | HydraPartialBuilds[T]``, both are which are now part of the public API of ``hydra_zen.typing``.

(See :pull:`186` and :pull:`230` for additional details)

Compatibility-Breaking Changes
------------------------------

``hydra_zen.typing.PartialBuilds`` is no longer a runtime-checkable protocol.
Code that used ``PartialBuilds`` in this way can be updated as follows:


+---------------------------------------------------+--------------------------------------------------------------------------+
|                                                   |                                                                          |
| .. code-block:: pycon                             | .. code-block:: pycon                                                    |
|    :caption: hydra-zen < 0.6.0                    |    :caption: 0.6.0 <= hydra-zen                                          |
|                                                   |                                                                          |
|    >>> from hydra_zen.typing import PartialBuilds |    >>> from hydra_zen.typing import HydraPartialBuilds, ZenPartialBuilds |
|                                                   |                                                                          |
|    >>> Conf = builds(int, zen_partial=True)       |    >>> Conf = builds(int, zen_partial=True)                              |
|    >>> isinstance(Conf, PartialBuilds)            |    >>> isinstance(Conf, (HydraPartialBuilds, ZenPartialBuilds))          |
|    True                                           |    True                                                                  |
+---------------------------------------------------+--------------------------------------------------------------------------+

.. _v0.5.0:

------------------
0.5.0 - 2022-01-27
------------------

This release primarily improves the ability of :func:`~hydra_zen.builds` to inspect and
the signatures of its targets; thus its ability to both auto-generate and validate 
configs is improved. This includes automatic support for specifying "partial'd" objects 
-- objects produced by :py:func:`functools.partial` -- as configured values, and even as
the target of :func:`~hydra_zen.builds`.

New Features
------------
- Objects produced by :py:func:`functools.partial` can now be specified directly as configured values in :func:`~hydra_zen.make_config` and :func:`~hydra_zen.builds`. See :pull:`198` for examples.
- An object produced by :py:func:`functools.partial` can now be specified as the target of :func:`~hydra_zen.builds`; ``builds`` will automatically "unpack" this partial'd object and incorporate its arguments into the config. See :pull:`199` for examples.

Improvements
------------
- Fixed an edge case `caused by an upstream bug in inspect.signature <https://bugs.python.org/issue40897>`_, which prevented :func:`~hydra_zen.builds` from accessing the appropriate signature for some target classes. This affected a couple of popular PyTorch classes, such as ``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``. See :pull:`189` for examples. 
- When appropriate, ``builds(<target>, ...)`` will now consult ``<target>.__new__`` to acquire the type-hints of the target's signature. See :pull:`189` for examples. 
- Fixed an edge case in the :ref:`type-widening behavior <type-support>` in both :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config` where a ``Builds``-like annotation would be widened to ``Any``; this widening was too aggressive. See :pull:`185` for examples.
- :ref:`Type widening <type-support>` will now be applied to configured fields where an interpolated variable -- a string of form ``"${<var-name>}"`` -- is specified. See :issue:`206` for rationale and examples.
- Fixed incomplete annotations for ``builds(..., zen_wrappers=<..>)``. See :pull:`180`

Compatibility-Breaking Changes
------------------------------

The deprecations :ref:`introduced in v0.3.0 <0p3p0-deprecations>` are now errors. Refer to those notes for details and for solutions for fixing stale code.


Notes
-----
It should be noted that the aforementioned improvements to :func:`~hydra_zen.builds` 
can change the interface to your app.

For instance, if you were configuring ``torch.utils.data.DataLoader``, note the 
following difference in behavior:

.. code-block:: python

   import torch as tr
   from hydra_zen import builds, to_yaml

   # DataLoader was affected by a bug in `inspect.signature`
   ConfLoader = builds(tr.utils.data.DataLoader, populate_full_signature=True)

Before 0.5.0:

.. code-block:: pycon

   >>> print(to_yaml(ConfLoader))  # builds could not access signature
   _target_: torch.utils.data.dataloader.DataLoader

After:

.. code-block:: pycon

   >>> print(to_yaml(ConfLoader))
   _target_: torch.utils.data.dataloader.DataLoader
   dataset: ???
   batch_size: 1
   shuffle: false
   sampler: null
   batch_sampler: null
   num_workers: 0
   collate_fn: null
   pin_memory: false
   drop_last: false
   timeout: 0.0
   worker_init_fn: null
   multiprocessing_context: null
   generator: null
   prefetch_factor: 2
   persistent_workers: false


.. _v0.4.1:

------------------
0.4.1 - 2021-12-06
------------------

:ref:`v0.4.0` introduced an undocumented, compatibility-breaking change to how hydra-zen treats :py:class:`enum.Enum` values. This patch reverts that change.

.. _v0.4.0:

------------------
0.4.0 - 2021-12-05
------------------

This release makes improvements to the validation performed by hydra-zen's 
:ref:`config-creation functions <create-config>`. It also adds specialized support for 
types that are not natively supported by Hydra.

Also included is an important compatibility-breaking change and a downstream 
fix for an upstream bug in 
`omegaconf <https://omegaconf.readthedocs.io/en/2.1_branch/>`_ (a library on which 
Hydra intimately depends). Thus it is highly recommended that users prioritize 
upgrading to hydra-zen v0.4.0.

New Features
------------

- Strict runtime *and* static validation of configuration types. See :pull:`163` for detailed descriptions and examples.
  
    hydra-zen's :ref:`config-creation functions <create-config>` now provide both strict runtime and static validation of the configured values that they are fed. Thus users will have a much easier time identifying and diagnosing bad configs, before launching a Hydra job.
- Specialized support for additional configuration-value types. See :pull:`163` for detailed descriptions and examples.

   Now values of types like :py:class:`complex` and :py:class:`pathlib.Path` can be specified directly in hydra-zen's configuration functions, and hydra-zen will automatically construct nested configs for those values. Consult :ref:`valid-types` for a complete list of the additional types that are supported.

Compatibility-Breaking Changes
------------------------------
We changed the behavior of :func:`~hydra_zen.builds` when 
`populate_full_signature=True` and one or more base-classes are specified for 
inheritance. 

Previously, fields specified by the parent class would take priority over those that 
would be auto-populated. However, this behavior is unintuitive as 
`populate_full_signature=True` should behave identically as the case where one 
manually-specifies the arguments from a target's signature. Thus we have changed the 
behavior accordingly. Please read more about it in :pull:`174`.

Bug Fixes
---------
The following bug was discovered in ``omegaconf <= 2.1.1``: a config that specifies a 
mutable default value for a field, but inherits from a parent that provides a 
non-mutable value for that field, will instantiate with the parent's field. Please read more about this issue, and our downstream fix for it, at :pull:`172`. 

It is recommended that users upgrade to the latest version of omegaconf once it is 
released, which will likely include a proper upstream fix of the bug.

Other improvements
------------------
hydra-zen will never be the first to import third-party libraries for which it provides 
specialized support (e.g., NumPy).

.. _v0.3.1:

------------------
0.3.1 - 2021-11-13
------------------

This release fixes a bug that was reported in :issue:`161`. Prior to this patch,
there was a bug in :func:`~hydra_zen.builds` where specifying ``populate_full_sig=True``
for a target that did not have ``**kwargs`` caused all user-specified zen-meta fields
to be excluded from the resulting config.

.. _v0.3.0:

------------------
0.3.0 - 2021-10-27
------------------

This release adds many new features to hydra-zen, and is a big step towards ``v1.0.0``. It also introduces some significant API changes, meaning that there are notable deprecations of expressions that were valid in ``v0.2.0``.

.. note::

   ???? We have completely rewritten our docs! The docs now follow the `Di??taxis Framework for technical documentation authoring <https://diataxis.fr/>`_.

.. admonition:: Join the Discussion ????

   The hydra-zen project `now has a discussion board <https://github.com/mit-ll-responsible-ai/hydra-zen/discussions>`_. Stop by and say "hi"! 


New Features
------------
- The introduction of ``builds(..., zen_wrappers=<>)``. 
  
    This is an extremely powerful feature that enables one to modify the instantiation of a builds-config, by including wrappers in a target's configuration. `Read more about it here <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/122>`_.
- Rich support for runtime type-checking of configurations. 

   Piggybacking off of the introduction of the ``zen_wrappers`` feature, **hydra-zen now offers support for customized runtime type-checking**. Presently, either of two type-checking libraries can be used: pydantic and beartype.

   - `Read about hydra-zen compatibility with pydantic <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/126>`_
   - `Read about hydra-zen compatibility with beartype <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/128>`_
   
  The type-checking capabilities offered by :func:`~hydra_zen.third_party.pydantic.validates_with_pydantic` and :func:`~hydra_zen.third_party.beartype.validates_with_beartype`, respectively, are both far more robust than those `offered by Hydra <https://hydra.cc/docs/next/tutorials/structured_config/intro/#structured-configs-supports>`_.
- A new, simplified method for creating a structured config, via :func:`~hydra_zen.make_config`.
  
   This serves as a much more succinct way to create a dataclass, where specifying type-annotations is optional. Additionally, provided type-annotations and default values are automatically adapted to be made compatible with Hydra. `Read more here <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/130>`_.
- :func:`~hydra_zen.make_custom_builds_fn`, which enables us to produce new "copies" of the :func:`~hydra_zen.builds` function, but with customized default-values.
- :func:`~hydra_zen.get_target`, which is used to retrieve target-objects from structured configs. See :pull:`94`
- ``builds(..., zen_meta=<dict>)`` users to attach "meta" fields to a targeted config, which will *not* be used by instantiate when building the target. 

   A meta-field can be referenced via relative interpolation; this
   interpolation will be valid no matter where the configuration is
   utilized. See :pull:`112`.

.. _0p3p0-deprecations:

Deprecations
------------
- The use of both ``hydra_zen.experimental.hydra_run`` and ``hydra_zen.experimental.hydra_multirun`` are deprecated in favor of the the function :func:`~hydra_zen.launch`.
- Creating partial configurations with ``builds(..., hydra_partial=True)`` is now deprecated in favor of ``builds(..., zen_partial=True)``.
- The first argument of :func:`~hydra_zen.builds` is now a positional-only argument. Code that specifies ``builds(target=<target>, ...)`` will now raise a deprecation warning; use ``builds(<target>, ...)`` instead. Previously, it was impossible to specify ``target`` as a keyword argument for the object being configured; now, e.g., ``builds(dict, target=1)`` will work. (See: `#104 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/104>`_).
- All keyword arguments of the form ``zen_xx``, ``hydra_xx``, and ``_zen_xx`` are reserved by both :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config`, to ensure that future features introduced by Hydra and hydra-zen will not cause compatibility conflicts for users.


Additional Items
----------------

- Improves type-annotations on :func:`~hydra_zen.builds`. Now, e.g., ``builds("hi")`` will be marked as invalid by static checkers (the target of :func:`~hydra_zen.builds` must be callable). See :pull:`104`.
- Migrates zen-specific fields to a new naming-scheme, and zen-specific processing to a universal mechanism. See :pull:`110` for more details.
- Ensures that hydra-zen's source code is "pyright-clean", under `pyright's basic type-checking mode <https://github.com/microsoft/pyright/blob/main/docs/configuration.md#diagnostic-rule-defaults>`_. `#101 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/101>`_
- Adds to all public modules/packages an ``__all__`` field. See :pull:`99`.
- Adds PEP 561 compliance (e.g. hydra-zen is now compatible with mypy). See :pull:`97`.
- Refactors hydra-zen's internals using `shed <https://pypi.org/project/shed/>`_. See :pull:`95`.
- Makes improvements to hydra-zen's test suite. See :pull:`90` and :pull:`91`.

.. _v0.2.0:

------------------
0.2.0 - 2021-08-12
------------------

This release:

- Improves hydra-zen's `automatic type refinement <https://mit-ll-responsible-ai.github.io/hydra-zen/structured_configs.html#automatic-type-refinement>`_. See :pull:`84` for details
- Cleans up the namespace of ```hydra_zen.typing``. See :pull:`85` for details.

**Compatibility-Breaking Changes**

- The protocol ``hydra_zen.typing.DataClass`` is no longer available in the public namespace, as it is not intended for public use. To continue using this protocol, you can import it from ``hydra_zen.typing._implementations``, but note that it is potentially subject to future changes or removal.


.. _v0.1.0:

------------------
0.1.0 - 2021-08-04
------------------

This is hydra-zen's first stable release on PyPI!
Although we have not yet released version `v1.0.0`, it should be noted that hydra-zen's codebase is thoroughly tested.
Its test suite makes keen use of the property-based testing library `Hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_.
Furthermore, 100% code coverage is enforced on all commits into `main`.

We plan to have an aggressive release schedule for compatibility-preserving patches of bug-fixes and quality-of-life improvements (e.g. improved type annotations).
hydra-zen will maintain a wide window of compatibility with Hydra versions; we test against pre-releases of Hydra and will maintain compatibility with its future releases.
