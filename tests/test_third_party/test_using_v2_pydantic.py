# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import sys

import pydantic
import pytest
from pydantic import BaseModel

if pydantic.__version__.startswith("1."):
    pytest.skip("These tests are for pydantic v2", allow_module_level=True)


def test_BaseModel():
    _pydantic = sys.modules.get("pydantic")
    assert _pydantic is not None
    assert _pydantic.BaseModel is BaseModel
