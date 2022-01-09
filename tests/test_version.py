# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


def test_version():
    import hydra_zen

    assert isinstance(hydra_zen.__version__, str)
    assert hydra_zen.__version__
    assert "unknown" not in hydra_zen.__version__
