# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import Any, Dict

import pytest

import hydra_zen
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
from hydra_zen.typing import DataclassOptions, ZenConvert
from tests.pyright_utils import (
    PYRIGHT_PATH,
    PyrightOutput,
    list_error_messages,
    pyright_analyze,
)

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
from hydra_zen.typing import ZenConvert, DataclassOptions
"""

# We use this to cache pyright scan results
# so that we can report their results via parameterized tests
PYRIGHT_SCAN_RESULTS: Dict[Any, PyrightOutput] = {}


FUNCS_TO_SCAN = [
    ZenField,
    ZenStore,
    builds.__call__,
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
    DataclassOptions,
]

if PYRIGHT_PATH is not None:
    for obj, scan in zip(
        FUNCS_TO_SCAN,
        pyright_analyze(
            *FUNCS_TO_SCAN,
            scan_docstring=True,
            report_unnecessary_type_ignore_comment=True,
            preamble=preamble,
        ),
    ):
        PYRIGHT_SCAN_RESULTS[obj] = scan


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
        DataclassOptions,
    ],
)
def test_docstrings_scan_clean_via_pyright(func):
    results = PYRIGHT_SCAN_RESULTS[func]
    assert results["summary"]["errorCount"] == 0, list_error_messages(results)


docs_src = Path(hydra_zen.__file__).parents[2] / "docs" / "source"


files = [
    pytest.param(f, {}, id=str(f.absolute()))
    for f in list(docs_src.glob("*.rst"))
    if f.name != "changes.rst"
]

files += [
    pytest.param(f, {"reportMissingImports": False}, id=str(f.absolute()))
    for f in list(docs_src.glob("tutorials/*.rst"))
]

files += [
    pytest.param(f, {"reportMissingImports": False}, id=str(f.absolute()))
    for f in list(docs_src.glob("how_to/*.rst"))
]

files += [
    pytest.param(f, {"reportMissingImports": False}, id=str(f.absolute()))
    for f in list(docs_src.glob("explanation/*.rst"))
]

if PYRIGHT_PATH is not None:
    raw_files = [f.values[0] for f in files]
    pyright_config = {k: v for d in [f.values[1] for f in files] for k, v in d.items()}

    for obj, scan in zip(
        raw_files,
        pyright_analyze(
            *raw_files,
            report_unnecessary_type_ignore_comment=True,
            preamble=preamble,
            pyright_config=pyright_config,
        ),
    ):
        PYRIGHT_SCAN_RESULTS[obj] = scan


@pytest.mark.skipif(PYRIGHT_PATH is None, reason="pyright is not installed")
@pytest.mark.skipif(not files, reason="docs not found")
@pytest.mark.parametrize(
    "func, pyright_config",
    files,
)
def test_rst_docs_scan_clean_via_pyright(func, pyright_config):
    results = PYRIGHT_SCAN_RESULTS[func]
    errors = [
        e
        for e in list_error_messages(results)
        if "is obscured by a declaration of the same name" not in e
    ]
    assert not errors
