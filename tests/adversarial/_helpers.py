"""Shared deterministic fixtures for adversarial TensorPoW tests."""

from __future__ import annotations

from tensorpow.chain.blocks import (
    Anchor,
    FeeFloorEntry,
    Fruit,
    anchor_reward_root,
    fee_floor_set_root,
    fruit_set_root,
    parent_candidate_root,
    tx_merkle_root,
)
from tensorpow.chain.headers import AnchorHeader, FruitHeader
from tensorpow.crypto.hash import DOMAIN_SHARD_TREE, HASH_LEN_BYTES, domain_hash
from tensorpow.crypto.signatures import SIG_TYPE_ED25519_BIT, sign
from tensorpow.genesis import GENESIS_CHAIN_ID_TESTNET, GenesisInputs, build_genesis_artifact
from tensorpow.mempool import ROOT_SHARD_ID, ShardTree
from tensorpow.pow.challenge import FORMAT_EPOCH
from tensorpow.state.utxo import TEMPLATE_PKH, UTXO, Outpoint
from tensorpow.tx.script import pubkey_hash
from tensorpow.tx.transaction import Input, Output, Transaction

PUBLIC_KEY = bytes.fromhex("343010a1aba8774dd1e6f4f0c3349bae6824908a1e64cd638dc2ed1bc625af1d")
PRIVATE_KEY = bytes.fromhex("cd4f7f79a2b8168f5cbeccb55d415492fd3504e52ed4fe7b02ea404fede9a40b")
OWNER_PUBKEY_HASH = pubkey_hash(PUBLIC_KEY)


def h(index: int) -> bytes:
    """Return a stable 32-byte test hash."""

    return index.to_bytes(HASH_LEN_BYTES, "big")


def outpoint(seed: int, output_index: int = 0) -> Outpoint:
    """Return a deterministic funded outpoint."""

    return Outpoint(h(seed), output_index)


def utxo(seed: int, *, amount: int, owner_pubkey_hash: bytes = OWNER_PUBKEY_HASH) -> UTXO:
    """Return a spendable PKH UTXO owned by the shared test key."""

    return UTXO(
        outpoint=outpoint(seed),
        amount_matoms=amount,
        template_id=TEMPLATE_PKH,
        owner_pubkey_hash=owner_pubkey_hash,
        payload=owner_pubkey_hash,
    )


def coinbase_tx(seed: int, *, amount: int = 1) -> Transaction:
    """Return a unique deterministic coinbase transaction."""

    return Transaction.coinbase(
        (
            Output(
                amount_matoms=amount,
                template_id=TEMPLATE_PKH,
                payload=h(10_000 + seed),
            ),
        )
    )


def signed_tx(
    spent: UTXO,
    *,
    fee: int,
    recipient_pubkey_hash: bytes = OWNER_PUBKEY_HASH,
    locktime_ms: int = 0,
    lockheight: int = 0,
) -> Transaction:
    """Build and sign a one-input, one-output transaction."""

    output = Output(
        amount_matoms=spent.amount_matoms - fee,
        template_id=TEMPLATE_PKH,
        payload=recipient_pubkey_hash,
    )
    unsigned = Transaction(
        version=FORMAT_EPOCH,
        sig_type=0,
        locktime_ms=locktime_ms,
        lockheight=lockheight,
        inputs=(Input(spent.outpoint),),
        outputs=(output,),
    )
    witness = sign(unsigned.sighash(0), PRIVATE_KEY) + PUBLIC_KEY
    return Transaction(
        version=unsigned.version,
        sig_type=unsigned.sig_type,
        locktime_ms=unsigned.locktime_ms,
        lockheight=unsigned.lockheight,
        inputs=(Input(spent.outpoint, witness=witness),),
        outputs=unsigned.outputs,
    )


def fruit(
    transactions: tuple[bytes, ...],
    *,
    nonce: int,
    timestamp_ms: int,
    parent_selected: bytes = bytes(HASH_LEN_BYTES),
    latest_anchor: bytes = bytes(HASH_LEN_BYTES),
    shard_id: int = ROOT_SHARD_ID,
) -> Fruit:
    """Return a canonical fruit for supplied canonical transaction bytes."""

    header = FruitHeader(
        version=FORMAT_EPOCH,
        sig_type_supported=SIG_TYPE_ED25519_BIT,
        parent_selected=parent_selected,
        parent_bitmap=b"",
        latest_anchor=latest_anchor,
        tx_merkle_root=tx_merkle_root(transactions),
        timestamp_ms=timestamp_ms,
        shard_id=shard_id,
        nonce=nonce,
    )
    return Fruit(header=header, transactions=transactions)


def anchor(
    *,
    shard_tree: ShardTree | None = None,
    shard_tree_bytes: bytes | None = None,
    fee_floor_matoms_per_kb: int = 0,
    covered_fruit_hashes: tuple[bytes, ...] = (),
    parent_anchor: bytes = bytes(HASH_LEN_BYTES),
    timestamp_ms: int = 1,
    nonce: int = 1,
) -> Anchor:
    """Return a canonical anchor, optionally with deliberately malformed tree bytes."""

    tree_bytes = ShardTree().serialize() if shard_tree is None else shard_tree.serialize()
    if shard_tree_bytes is not None:
        tree_bytes = shard_tree_bytes
    fee_entries = (
        (FeeFloorEntry(ROOT_SHARD_ID, fee_floor_matoms_per_kb),)
        if shard_tree_bytes is not None
        else tuple(
            FeeFloorEntry(shard_id, fee_floor_matoms_per_kb)
            for shard_id in (ShardTree() if shard_tree is None else shard_tree).leaf_shard_ids
        )
    )
    covered = tuple(sorted(covered_fruit_hashes))
    parent_candidates = covered
    header = AnchorHeader(
        version=FORMAT_EPOCH,
        parent_anchor=parent_anchor,
        fruit_set_root=fruit_set_root(covered),
        parent_candidate_root=parent_candidate_root(parent_candidates),
        shard_tree_state_root=domain_hash(DOMAIN_SHARD_TREE, tree_bytes),
        fee_floor_set_root=fee_floor_set_root(fee_entries),
        anchor_reward_root=anchor_reward_root(()),
        timestamp_ms=timestamp_ms,
        nonce=nonce,
    )
    return Anchor(
        header=header,
        covered_fruit_hashes=covered,
        parent_candidate_hashes=parent_candidates,
        shard_tree_bytes=tree_bytes,
        fee_floor_entries=fee_entries,
        genesis_commitment=h(50_000 + nonce) if not covered else bytes(HASH_LEN_BYTES),
    )


def genesis_anchor() -> Anchor:
    """Return the deterministic adversarial-test genesis anchor."""

    return build_genesis_artifact(
        GenesisInputs.create(
            chain_id=GENESIS_CHAIN_ID_TESTNET,
            whitepaper_hash=h(60_000),
            bitcoin_block_hash=h(60_001),
            ethereum_block_hash=h(60_002),
            founder_pubkey_hash=h(60_003),
        )
    ).anchor
