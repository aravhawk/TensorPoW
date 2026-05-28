"""Tests for per-shard mempool admission and fee floors."""

from __future__ import annotations

from tensorpow.crypto.signatures import sign
from tensorpow.mempool import (
    FEE_FLOOR_WINDOW_FRUITS,
    MAX_FRUIT_PAYLOAD_BYTES,
    ROOT_SHARD_ID,
    Mempool,
    ShardTree,
    apply_split,
)
from tensorpow.mempool.mempool import MempoolEntry, _fee_rate_matoms_per_kb, _selection_key
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint, UTXOSet
from tensorpow.tx.script import pubkey_hash
from tensorpow.tx.transaction import FORMAT_EPOCH, Input, Output, Transaction

PUB1 = bytes.fromhex("343010a1aba8774dd1e6f4f0c3349bae6824908a1e64cd638dc2ed1bc625af1d")
PRIV1 = bytes.fromhex("cd4f7f79a2b8168f5cbeccb55d415492fd3504e52ed4fe7b02ea404fede9a40b")
PKH1 = pubkey_hash(PUB1)


def test_add_duplicate_remove_and_evict_below_floor() -> None:
    utxo = _utxo(1, amount=1_000)
    tx = _signed_tx(utxo, fee=200)
    mempool = Mempool(utxo_view=UTXOSet((utxo,)))

    result = mempool.add_tx(tx)

    assert result.accepted
    assert result.tx_id == tx.tx_id()
    assert result.shard_id == ROOT_SHARD_ID
    assert mempool.contains(tx.tx_id())
    assert mempool.get(tx.tx_id()) == tx
    assert mempool.add_tx(tx).reason == "duplicate"

    entry = mempool.get_entry(tx.tx_id())
    assert entry is not None
    evicted = mempool.evict_below_floor(ROOT_SHARD_ID, entry.fee_rate_matoms_per_kb + 1)

    assert evicted == (tx.tx_id(),)
    assert not mempool.contains(tx.tx_id())
    assert mempool.remove(tx.tx_id()) is None


def test_select_for_fruit_uses_fee_priority_tx_id_tiebreaker_and_payload_budget() -> None:
    high_utxo = _utxo(2, amount=2_000)
    tie_a_utxo = _utxo(3, amount=2_000)
    tie_b_utxo = _utxo(4, amount=2_000)
    high_fee = _signed_tx(high_utxo, fee=700)
    tie_a = _signed_tx(tie_a_utxo, fee=300)
    tie_b = _signed_tx(tie_b_utxo, fee=300)
    mempool = Mempool(utxo_view=UTXOSet((high_utxo, tie_a_utxo, tie_b_utxo)))

    assert mempool.add_tx(tie_a)
    assert mempool.add_tx(high_fee)
    assert mempool.add_tx(tie_b)

    selected = mempool.select_for_fruit(ROOT_SHARD_ID)
    tie_order = sorted((tie_a, tie_b), key=lambda tx: tx.tx_id())

    assert selected == [high_fee, *tie_order]

    payload_budget = len(high_fee.to_bytes())
    limited = mempool.select_for_fruit(ROOT_SHARD_ID, payload_budget=payload_budget)

    assert limited == [high_fee]
    assert sum(len(tx.to_bytes()) for tx in limited) <= payload_budget


def test_selection_key_uses_fee_rate_before_burned_tip_rate_under_nonzero_floor() -> None:
    high_fee_low_tip = _entry_for_selection(
        fee_matoms=2,
        tx_size_bytes=501,
        tx_id=b"\x02" * 32,
    )
    lower_fee_higher_tip = _entry_for_selection(
        fee_matoms=1,
        tx_size_bytes=334,
        tx_id=b"\x01" * 32,
    )

    assert high_fee_low_tip.fee_rate_matoms_per_kb > lower_fee_higher_tip.fee_rate_matoms_per_kb
    assert sorted((lower_fee_higher_tip, high_fee_low_tip), key=_selection_key) == [
        high_fee_low_tip,
        lower_fee_higher_tip,
    ]


def test_fee_floor_rises_under_heavy_load_and_falls_under_light_load() -> None:
    mempool = Mempool()

    assert mempool.fee_floor(ROOT_SHARD_ID) == 0
    first_floor = mempool.record_confirmed_fruit(
        ROOT_SHARD_ID,
        floor_eligible_fees_matoms=MAX_FRUIT_PAYLOAD_BYTES,
        payload_bytes=MAX_FRUIT_PAYLOAD_BYTES,
    )
    for _ in range(7):
        mempool.record_confirmed_fruit(
            ROOT_SHARD_ID,
            floor_eligible_fees_matoms=MAX_FRUIT_PAYLOAD_BYTES,
            payload_bytes=MAX_FRUIT_PAYLOAD_BYTES,
        )
    heavy_floor = mempool.fee_floor(ROOT_SHARD_ID)

    assert first_floor > 0
    assert heavy_floor > first_floor

    for _ in range(FEE_FLOOR_WINDOW_FRUITS):
        mempool.record_confirmed_fruit(
            ROOT_SHARD_ID,
            floor_eligible_fees_matoms=0,
            payload_bytes=MAX_FRUIT_PAYLOAD_BYTES,
        )

    assert mempool.recent_fee_rate(ROOT_SHARD_ID) == 0
    assert mempool.fee_floor(ROOT_SHARD_ID) < heavy_floor


def test_rejects_malformed_and_wrong_shard_transactions() -> None:
    utxo = _utxo(5, amount=1_000)
    tx = _signed_tx(utxo, fee=100)
    tree = apply_split(ShardTree(), ROOT_SHARD_ID)
    wrong_shard_id = next(
        shard_id for shard_id in tree.leaf_shard_ids if shard_id != tree.route_tx(tx.tx_id())
    )
    mempool = Mempool(shard_tree=tree, utxo_view=UTXOSet((utxo,)))

    assert mempool.add_tx(b"\x00").reason == "malformed"
    assert mempool.add_tx(tx, shard_id=wrong_shard_id).reason == "wrong_shard"


def test_rejects_negative_fee_below_floor_and_script_failure() -> None:
    negative_fee_utxo = _utxo(6, amount=1_000)
    below_floor_utxo = _utxo(7, amount=1_000)
    bad_script_utxo = _utxo(8, amount=1_000)
    view = UTXOSet((negative_fee_utxo, below_floor_utxo, bad_script_utxo))

    negative_fee_tx = _signed_tx(negative_fee_utxo, fee=-1)
    mempool = Mempool(utxo_view=view)
    assert mempool.add_tx(negative_fee_tx).reason == "negative_fee"

    below_floor_tx = _signed_tx(below_floor_utxo, fee=100)
    below_floor_entry_rate = 100 * 1000 // len(below_floor_tx.to_bytes())
    mempool.set_fee_floor(ROOT_SHARD_ID, below_floor_entry_rate + 1)
    assert mempool.add_tx(below_floor_tx).reason == "below_fee_floor"

    bad_script_tx = _signed_tx(bad_script_utxo, fee=100, corrupt_witness=True)
    assert mempool.add_tx(bad_script_tx).reason == "script_failed"


def test_rejects_missing_utxo_and_mempool_input_conflicts() -> None:
    utxo = _utxo(9, amount=1_000)
    tx = _signed_tx(utxo, fee=100)

    assert Mempool().add_tx(tx).reason == "missing_utxo_view"
    assert Mempool(utxo_view=UTXOSet()).add_tx(tx).reason == "missing_input"

    mempool = Mempool(utxo_view=UTXOSet((utxo,)))
    conflict = _signed_tx(utxo, fee=200)

    assert mempool.add_tx(tx)
    assert mempool.add_tx(conflict).reason == "conflict"


def test_rejects_unmatured_transaction_locktime_and_lockheight() -> None:
    utxo = _utxo(10, amount=1_000)
    tx = _signed_tx(utxo, fee=100, locktime_ms=100, lockheight=5)
    mempool = Mempool(utxo_view=UTXOSet((utxo,)))

    assert mempool.add_tx(tx, current_time_ms=99, current_height=5).reason == "tx_lock_unmatured"
    assert mempool.add_tx(tx, current_time_ms=100, current_height=4).reason == "tx_lock_unmatured"
    assert mempool.add_tx(tx, current_time_ms=100, current_height=5).accepted


def test_coinbase_bypasses_fee_floor_without_utxo_view() -> None:
    tx = Transaction.coinbase((Output(1, TEMPLATE_PKH, payload=PKH1),))
    mempool = Mempool()
    mempool.set_fee_floor(ROOT_SHARD_ID, 1_000_000)

    result = mempool.add_tx(tx)

    assert result.accepted
    assert mempool.select_for_fruit(ROOT_SHARD_ID) == [tx]


def _outpoint(seed: int) -> Outpoint:
    return Outpoint(bytes([seed]) * 32, 0)


def _utxo(seed: int, *, amount: int) -> UTXO:
    return UTXO(
        outpoint=_outpoint(seed),
        amount_matoms=amount,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=PKH1,
    )


def _signed_tx(
    utxo: UTXO,
    *,
    fee: int,
    corrupt_witness: bool = False,
    locktime_ms: int = 0,
    lockheight: int = 0,
) -> Transaction:
    output = Output(utxo.amount_matoms - fee, TEMPLATE_PKH, payload=PKH1)
    unsigned_input = Input(utxo.outpoint)
    unsigned_tx = Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=locktime_ms,
        lockheight=lockheight,
        inputs=(unsigned_input,),
        outputs=(output,),
    )
    signature = sign(unsigned_tx.sighash(0), PRIV1)
    if corrupt_witness:
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    return Transaction(
        version=unsigned_tx.version,
        sig_type=unsigned_tx.sig_type,
        locktime_ms=unsigned_tx.locktime_ms,
        lockheight=unsigned_tx.lockheight,
        inputs=(Input(utxo.outpoint, witness=signature + PUB1),),
        outputs=unsigned_tx.outputs,
    )


def _entry_for_selection(*, fee_matoms: int, tx_size_bytes: int, tx_id: bytes) -> MempoolEntry:
    return MempoolEntry(
        tx=_signed_tx(_utxo(tx_id[0], amount=1_000), fee=100),
        tx_id=tx_id,
        shard_id=ROOT_SHARD_ID,
        tx_size_bytes=tx_size_bytes,
        fee_matoms=fee_matoms,
        fee_rate_matoms_per_kb=_fee_rate_matoms_per_kb(fee_matoms, tx_size_bytes),
        is_coinbase=False,
    )
