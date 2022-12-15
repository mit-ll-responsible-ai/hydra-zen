# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import sys
from datetime import datetime
from pathlib import Path

import pytest
from pytest import param

import hydra_zen


def test_version():
    assert isinstance(hydra_zen.__version__, str)
    assert hydra_zen.__version__
    assert "unknown" not in hydra_zen.__version__


root = Path(hydra_zen.__file__).parent.parent

expected_header = f"""
# Copyright (c) {datetime.now().year} Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
""".lstrip()

src_files = sorted(root.glob("hydra_zen/**/*.py"))
test_files = sorted(root.parent.glob("tests/**/*.py"))

assert src_files
assert test_files


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 9),
    reason="Only test project structure for one version.",
)
@pytest.mark.parametrize("file", [param(f, id=str(f)) for f in src_files + test_files])
def test_file_header(file: Path):
    src = file.read_text()[: len(expected_header)]
    if file.name == "__init__.py" and not src:
        pytest.skip(reason="Empty __init__.py file doesn't need header.")

    if file.name == "_version.py":
        pytest.skip(reason="scm_setuptools file doesn't need header.")

    assert src == expected_header
