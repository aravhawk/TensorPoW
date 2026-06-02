"""Regression tests for adversarial fixture helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tests.adversarial._helpers import coinbase_tx, fruit


def test_fruit_helper_requires_explicit_parent() -> None:
    fruit_without_type_check = cast(Any, fruit)

    with pytest.raises(TypeError, match="parent_selected"):
        fruit_without_type_check(
            (coinbase_tx(901).to_bytes(),),
            nonce=901,
            timestamp_ms=1,
        )
