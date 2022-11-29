# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import pytest

from hydra_zen import (
    ZenField,
    ZenStore,
    builds,
    get_target,
    hydrated_dataclass,
    instantiate,
    is_partial_builds,
    just,
    load_from_yaml,
    make_config,
    make_custom_builds_fn,
    save_as_yaml,
    to_yaml,
    uses_zen_processing,
    zen,
)
from hydra_zen.typing import ZenConvert
from tests.pyright_utils import PYRIGHT_PATH, list_error_messages, pyright_analyze

preamble = """from hydra_zen import (
    ZenField,
    ZenStore,
    builds,
    get_target,
    hydrated_dataclass,
    instantiate,
    is_partial_builds,
    just,
    launch,
    load_from_yaml,
    make_config,
    make_custom_builds_fn,
    save_as_yaml,
    to_yaml,
    uses_zen_processing,
    zen,
)
from hydra_zen.typing import ZenConvert
"""


@pytest.mark.skipif(PYRIGHT_PATH is None, reason="pyright is not installed")
@pytest.mark.parametrize(
    "func",
    [
        ZenField,
        ZenStore,
        builds,
        get_target,
        hydrated_dataclass,
        instantiate,
        is_partial_builds,
        just,
        # launch,  # TODO: add after https://github.com/mit-ll-responsible-ai/hydra-zen/pull/313 is merged
        load_from_yaml,
        make_config,
        make_custom_builds_fn,
        save_as_yaml,
        to_yaml,
        uses_zen_processing,
        zen,
        ZenConvert,
    ],
)
def test_docstrings_scan_clean_via_pyright(func):
    results = pyright_analyze(
        func,
        scan_docstring=True,
        report_unnecessary_type_ignore_comment=True,
        preamble=preamble,
    )
    assert results["summary"]["errorCount"] == 0, list_error_messages(results)
