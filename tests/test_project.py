# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import hydra_zen


def test_version():
    assert isinstance(hydra_zen.__version__, str)
    assert hydra_zen.__version__
    assert "unknown" not in hydra_zen.__version__
