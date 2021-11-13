=========
Changelog
=========

This is a record of all past hydra-zen releases and what went into them, in reverse chronological order.
All previous releases should still be available on pip.

.. _v0.3.1:

------------------
0.3.1 - 2021-11-13
------------------

The release fixes a bug that was reported in :issue:`161`. Prior to this patch,
there was a bug in :func:`~hydra_zen.builds` where specifying ``populate_full_sig=True``
for a target that did not have ``**kwargs`` caused all user-specified zen-meta fields
to be excluded from the resulting config.

.. _v0.3.0:

------------------
0.3.0 - 2021-10-27
------------------

This release adds many new features to hydra-zen, and is a big step towards ``v1.0.0``. It also introduces some significant API changes, meaning that there are notable deprecations of expressions that were valid in ``v0.2.0``.

.. note::

   📚 We have completely rewritten our docs! The docs now follow the `Diátaxis Framework for technical documentation authoring <https://diataxis.fr/>`_.

.. admonition:: Join the Discussion 💬

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
