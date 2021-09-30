=========
Changelog
=========

This is a record of all past hydra-zen releases and what went into them, in reverse chronological order.
All previous releases should still be available on pip.

.. _v0.3.0:

---------------------
0.3.0rc3 - 2021-09-20
---------------------

This release:

- Makes the "target" of `builds` a positional-only argument. Code that specifies ``builds(target=<target>, ...)`` will now raise a deprecation warning. Now, e.g., ``builds(dict, target=1)`` will work. `#104 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/104>`_
- Adds the ability to set "meta" fields via `builds`. See `#112 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/112>`_
- Adds the :func:`~hydra_zen.get_target` function for retrieving target-objects from structured configs. `#94 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/94>`_
- Improves type-annotations on `builds`. Now, e.g., ``builds("hi")`` will be marked as invalid by static checkers (the target of `builds` must be callable). `#104 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/104>`_.
- Migrates zen-specific fields to a new naming-scheme, and zen-specific processing to a universal mechanism. See `#110 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/110>`_ for more details.
- Ensures that hydra-zen's source code is "pyright-clean", under `pyright's basic type-checking mode <https://github.com/microsoft/pyright/blob/main/docs/configuration.md#diagnostic-rule-defaults>`_. `#101 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/101>`_
- Adds to all public modules/packages an ``__all__`` field. `#99 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/99>`_
- Adds PEP 561 compliance (e.g. hydra-zen is now compatible with mypy). `#97 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/97>`_
- Refactors hydra-zen's internals using `shed <https://pypi.org/project/shed/>`_. `#95 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/95>`_
- Makes improvements to hydra-zen's test suite. `#90 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/90>`_ and `#91 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/91>`_.

.. _v0.2.0:

------------------
0.2.0 - 2021-08-12
------------------

This release:

- Improves hydra-zen's `automatic type refinement <https://mit-ll-responsible-ai.github.io/hydra-zen/structured_configs.html#automatic-type-refinement>`_. See `#84 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/84>`_ for details
- Cleans up the namespace of ```hydra_zen.typing``. See `#85 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/85>`_ for details

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
