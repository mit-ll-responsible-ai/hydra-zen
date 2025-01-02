# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# flake8: noqa

import pytest

from tests.pyright_utils import rst_to_code

src1 = """.. tab-set::

   .. tab-item:: dataclass-based config

      .. code-block:: python
         :caption: Manually configuring ``DNN`` with a dataclass
      
         from dataclasses import dataclass
      
         @dataclass
         class Builds_DNN:
             input_size: int
             output_size: int
             layer_widths: Tuple[int, ...] = (5, 10, 5)
             device: str = "cpu"
             _target_: str = "vision.model.DNN"

   .. tab-item:: yaml-based config

      .. code-block:: yaml
         :caption: Manually configuring ``DNN`` with a YAML file
      
         _target_: vision.model.DNN
         input_size: ???
         output_size: ???
         layer_widths:
         - 5
         - 10
         - 5
"""
expected1 = """
from dataclasses import dataclass

@dataclass
class Builds_DNN:
    input_size: int
    output_size: int
    layer_widths: Tuple[int, ...] = (5, 10, 5)
    device: str = "cpu"
    _target_: str = "vision.model.DNN"

"""

src2 = """
hi
.. code-block:: pycon

    >>> from pathlib import Path 
    >>> def print_file(x: int):
    ...     with x.open("r") as f: 
    ...         print(f.read())
hi
hello
.. code-block:: shell

    blah blah
"""

src3 = """
hi
.. code-block:: python

    from pathlib import Path 

    def print_file(x: int):
        with x.open("r") as f: 
            print(f.read())
hi
hello

.. code-block:: shell

    blah blah
a
"""

expected2 = """from pathlib import Path 

def print_file(x: int):
    with x.open("r") as f: 
        print(f.read())
"""
expected3 = expected2


def strip_interspacing(x: str):
    return "\n".join(s for s in x.splitlines() if s)


@pytest.mark.parametrize(
    "src,expected",
    [
        pytest.param(src1, expected1, id="src1"),
        pytest.param(src2, expected2, id="src2"),
        pytest.param(src2 * 3, "\n".join([expected2] * 3), id="src2 repeat"),
        pytest.param(src3 * 5, "\n".join([expected3] * 5), id="src3 repeat"),
    ],
)
def test_rst_parsing(src: str, expected: str):
    assert strip_interspacing(rst_to_code(src)) == strip_interspacing(expected)
