def test_version():
    import hydra_utils

    assert isinstance(hydra_utils.__version__, str)
    assert hydra_utils.__version__
    assert "unknown" not in hydra_utils.__version__
