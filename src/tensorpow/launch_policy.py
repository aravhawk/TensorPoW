"""Public launch-gate policy constants."""

from __future__ import annotations

from typing import Final

PUBLIC_TESTNET_MIN_DAYS: Final[int] = 30
PUBLIC_TESTNET_MIN_NODES: Final[int] = 100
GENESIS_BTC_MIN_CONFIRMATIONS: Final[int] = 6
GENESIS_BITCOIN_SELECTION_RULE: Final[str] = "latest pre-ceremony block with 6 confirmations"
GENESIS_ETHEREUM_SELECTION_RULE: Final[str] = "latest finalized pre-ceremony block"

__all__ = [
    "GENESIS_BITCOIN_SELECTION_RULE",
    "GENESIS_BTC_MIN_CONFIRMATIONS",
    "GENESIS_ETHEREUM_SELECTION_RULE",
    "PUBLIC_TESTNET_MIN_DAYS",
    "PUBLIC_TESTNET_MIN_NODES",
]
