import pytest


def test_experimental_coerce_not_available():
    with pytest.raises(NotImplementedError):
        import hydra_zen.experimental.coerce as coerce

        type(coerce)  # touch to keep formatters from freaking out


def test_beartype_not_available():
    with pytest.raises(NotImplementedError):
        from hydra_zen.experimental.third_party.beartype import validates_with_beartype

        type(validates_with_beartype)  # touch to keep formatters from freaking out
