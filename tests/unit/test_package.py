"""Package-level smoke tests."""

import tensorpow


def test_package_exposes_version() -> None:
    assert tensorpow.__version__ == "0.1.0"
