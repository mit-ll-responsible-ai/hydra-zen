from datetime import datetime
from pathlib import Path

import pytest
from pytest import param

root = Path(__file__).parent.parent

expected_header = f"""
# Copyright (c) {datetime.now().year} Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
""".lstrip()

src_files = sorted(root.glob("src/hydra_zen/**/*.py"))
test_files = sorted(root.glob("tests/**/*.py"))

assert src_files
assert test_files


@pytest.mark.parametrize("file", [param(f, id=str(f)) for f in src_files + test_files])
def test_file_header(file: Path):
    src = file.read_text()[: len(expected_header)]
    if file.name == "__init__.py" and not src:
        pytest.skip(reason="Empty __init__.py file doesn't need header.")

    if file.name == "_version.py":
        pytest.skip(reason="scm_setuptools file doesn't need header.")

    assert src == expected_header
