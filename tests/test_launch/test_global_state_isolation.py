# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import os
import textwrap

import tests

# `pytester` is enabled via `pytest_plugins = "pytester"` in tests/conftest.py

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(tests.__file__)))


def test_clean_store_preserves_hydra_plugin_configs(pytester):
    """Regression test for https://github.com/mit-ll-responsible-ai/hydra-zen/issues/574

    The `clean_store` fixture snapshots and restores Hydra's global ConfigStore.
    Hydra registers its plugin-provided configs (e.g. ``hydra/sweeper/basic``)
    into that store lazily, upon first plugin discovery. If the fixture snapshots
    the store *before* discovery occurs and then restores that snapshot, it
    strips those plugin configs from the global store -- breaking any later test
    that composes the Hydra config (e.g. with ``return_hydra_config=True``) with
    ``MissingConfigException: Could not find 'hydra/sweeper/basic'``.

    This bug is order-dependent (hence the flakiness seen across Python versions
    in CI). Here we reproduce the triggering order deterministically in an
    isolated pytest subprocess that uses the *real* ``clean_store`` fixture: the
    first test snapshots via ``clean_store`` and then triggers discovery; the
    second test then requires ``hydra/sweeper/basic``.
    """
    # Make the project's real `clean_store` fixture available in the subprocess.
    pytester.makeconftest(
        f"""
        import sys
        sys.path.insert(0, {_REPO_ROOT!r})
        from tests.conftest import clean_store  # noqa: F401
        """
    )
    pytester.makepyfile(
        textwrap.dedent(
            """
            import pytest
            from hydra import compose, initialize
            from hydra.core.config_store import ConfigStore


            @pytest.mark.usefixtures("clean_store")
            def test_snapshot_then_trigger_discovery():
                # clean_store snapshots the ConfigStore here; this compose then
                # triggers Hydra's lazy plugin discovery.
                cs = ConfigStore.instance()
                cs.store(name="tmp", node={"a": 1})
                with initialize(config_path=None, version_base="1.3"):
                    compose(config_name="tmp", overrides=[])


            def test_requires_hydra_sweeper_basic():
                # Fails with MissingConfigException if the previous test's
                # clean_store teardown stripped hydra/sweeper/basic.
                cs = ConfigStore.instance()
                cs.store(name="tmp2", node={"a": 1})
                with initialize(config_path=None, version_base="1.3"):
                    compose(
                        config_name="tmp2", overrides=[], return_hydra_config=True
                    )
            """
        )
    )
    result = pytester.runpytest_subprocess("-p", "no:randomly")
    result.assert_outcomes(passed=2)
