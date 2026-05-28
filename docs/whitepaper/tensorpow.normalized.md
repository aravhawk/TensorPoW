# TensorPoW: A Tensor-Native Proof-of-Work Currency

Author: Arav Jain (@aravhawk)
Draft date: 2026-05-27

## Abstract

TensorPoW is a permissionless proof-of-work currency whose work function is
deterministic signed INT8 matrix multiplication with INT32 accumulation. The
chain separates high-rate transaction carriers, called fruits, from lower-rate
ordering checkpoints, called anchors. Fruits carry transactions under a
permanent work target. Anchors commit fruit sets, parent candidates, shard-tree
state, and fee-floor state under a WTEMA-adjusted target. The ledger is a pure
UTXO system: balances are derived only from unspent outputs, and genesis creates
no spendable allocation.

The design goal is to make useful parallel tensor hardware the natural mining
substrate while keeping consensus byte-exact across CUDA, MPS, x86 CPU, and ARM
CPU implementations. All consensus hashes use BLAKE3 with explicit domain
separation, all consensus integers are little-endian, and every consensus object
has a canonical byte encoding.

## 1. Introduction

Bitcoin demonstrated that an open network can order payments by making history
expensive to rewrite. TensorPoW keeps that central idea but changes the work
primitive and the block structure. Modern compute supply is increasingly shaped
by tensor accelerators. TensorPoW therefore uses a fixed 1024 by 1024 signed
INT8 matrix multiplication as its proof-of-work kernel. The output is serialized
as signed little-endian INT32 cells and hashed with BLAKE3.

TensorPoW is also a BlockDAG rather than a single-parent chain. Fruits can
reference multiple recent parents so honest work produced during propagation
races remains useful. Anchors periodically commit ordered fruit sets and the
state needed by wallets, miners, and light clients. The currency unit is TSC;
the atomic subunit is the matom, with 100,000,000 matoms per TSC.

This paper describes the permanent protocol semantics. It does not introduce a
numbered-generation network. Future incompatible rule changes require planned
hardforks of the same chain.

## 2. System Model

Nodes communicate over libp2p using TCP and QUIC transports, Noise security,
yamux multiplexing, Kademlia peer discovery, and Gossipsub 2.0 topic gossip.
The protocol defines gossip topics for fruits, anchors, and shard transactions.
Every wire message begins with the TensorPoW wire magic, a typed payload length,
the canonical payload, and a short BLAKE3-derived checksum. A malformed network
message is dropped before its payload can affect consensus.

The ledger is a pure UTXO ledger. There are no accounts, account nonces, smart
contract storage slots, or shared mutable balances. A transaction consumes
existing outpoints and creates new outputs. A transaction is valid only when the
referenced outputs exist, are unspent, satisfy their script templates, and obey
the active shard fee floor.

The network assumes asynchronous propagation and adversarial peers. It does not
enforce a consensus bandwidth minimum. Public routing is expected to work best
for operators that provision at least the recommended 10 MB/s bandwidth.

## 3. Cryptographic Commitments

Every consensus hash is BLAKE3 with a 32-byte output unless the protocol
explicitly calls for BLAKE3 XOF. Hash preimages begin with a one-byte domain
constant followed by canonical bytes. Domain separation is used for PoW
challenges, PoW outputs, fruit headers, anchor headers, transaction IDs,
signature hashes, outpoints, UTXO values, Merkle nodes, addresses, genesis,
shard trees, fee floors, and data-availability samples.

Ed25519 is the only active signature type at genesis. Transaction inputs carry a
signature type, and verifiers dispatch through a fixed signature table. Unknown
signature types fail validation. The reserved post-quantum signature identifier
does not validate until a hardfork activates rules for it.

Tensorcoin addresses are lowercase Bech32m strings with the human-readable part
`tsc`. The payload is `BLAKE3(DOMAIN_ADDRESS || ed25519_public_key)`, converted
from 8-bit bytes to 5-bit Bech32m words. Mixed case, wrong HRP, invalid
checksums, invalid padding, and wrong decoded payload lengths are invalid.

## 4. Tensor Proof of Work

The PoW kernel is:

```text
C[row, col] = sum(A[row, k] * B[k, col]) for k in [0, 1024)
```

`A` and `B` are signed INT8 matrices. `C` is a signed INT32 matrix. The
accumulator must not wrap, and floating-point promotion is not a valid
consensus implementation. Canonical output bytes serialize every INT32 cell in
row-major order as four signed little-endian bytes. The PoW digest is:

```text
pow_digest = BLAKE3(DOMAIN_POW_OUTPUT || canonical_output_bytes)
```

Targets and PoW digests are interpreted as unsigned little-endian 256-bit
integers. A proof passes when `int_le(pow_digest) <= int_le(target)`.

Challenge matrices are generated deterministically from the candidate header.
Fruit challenges include the format epoch, supported signature bitmask,
effective parent hashes, latest anchor, transaction Merkle root, timestamp,
shard id, and nonce. Anchor challenges include the parent anchor, fruit set
root, parent candidate root, shard-tree root, fee-floor root, anchor reward
root, timestamp, and nonce. The matrix bytes are read from BLAKE3 XOF streams,
one stream for `A` and one for `B`, and each byte maps to signed INT8 by
subtracting 256 when the byte is at least 128.

Fruit work uses a permanent target derived from 6,000,000,000 operations and
the fixed 1024 by 1024 matrix multiplication cost. Anchor genesis work starts at
1000 times the fruit work. Only anchor targets adjust over time, using WTEMA and
a 60-second interval target.

## 5. Fruits, Anchors, and Ordering

TensorPoW has two block classes.

Fruits carry transactions. A fruit header commits to its selected parent, a
bitmap of additional parents selected from the latest anchor's parent candidate
list, the latest anchor, the transaction Merkle root, timestamp, shard id, and
nonce. A fruit body must include a coinbase transaction first and may then carry
non-coinbase transactions for its shard. The aggregate fruit payload is capped
at 8 KB.

Anchors commit ordering and global interval state. An anchor header commits to
its parent anchor, the covered fruit set, the parent candidate list, the shard
tree, the fee-floor set, and anchor reward outputs. An anchor body carries
sorted covered fruit hashes, ordered parent candidates, canonical shard-tree
bytes, sorted fee-floor entries, anchor reward outputs, and the genesis
commitment only for the genesis anchor.

Fruit ordering uses a GHOSTDAG-style greedy algorithm. The selected parent is
the parent with the highest accumulated blue work, with lexicographic fruit hash
as the tie-breaker. A dynamic `k` parameter is computed from observed fruit
rate, observed propagation delay, and bounded protocol constants. Candidates
whose anticone size exceeds `k` are red. Topological order sorts by blue score,
timestamp, and fruit hash.

Finality is expressed in tiers. A transaction is Seen when it appears in a
valid known fruit, Fast at the fast blue-depth threshold, Economic at the
economic blue-depth threshold, Settlement at the settlement blue-depth and
anchor-depth thresholds, and AnchorSecured after anchor commitment depth.
Wall-clock finality estimates are advisory and do not affect validity.

## 6. Transactions, Scripts, and UTXO State

A transaction ID is `BLAKE3(DOMAIN_TX_ID || canonical_tx_bytes)`. A signature
hash is `BLAKE3(DOMAIN_TX_SIGHASH || canonical_tx_bytes_with_empty_witnesses ||
input_index_le)`. Transaction size is capped at 8 KB.

Transaction inputs reference 36-byte outpoints: a 32-byte transaction ID and a
4-byte little-endian output index. Outputs carry a matom amount, template id,
wall-time lock, height lock, and bounded template payload. Coinbase
transactions have zero inputs, must be first in a fruit, and must not mint more
than the assigned subsidy plus tips.

Active output templates are:

- `TEMPLATE_PKH`, whose payload is an owner public-key hash and whose witness is
  an Ed25519 signature plus public key.
- `TEMPLATE_MULTISIG`, whose payload is a threshold and bounded public-key list.
- `TEMPLATE_HASHLOCK`, whose witness reveals a BLAKE3 preimage before
  satisfying an inner template.

The UTXO root is a compact sparse Merkle tree keyed by
`BLAKE3(DOMAIN_OUTPOINT || outpoint_bytes)`. Leaves commit to both the outpoint
key and the canonical UTXO bytes. Empty subtrees have deterministic hashes by
depth. Inclusion and non-inclusion proofs are valid only when they reconstruct
the committed root from canonical node encodings.

## 7. Sharding and Fee Floors

The mempool is a binary shard tree with maximum depth 16. A shard id stores its
depth in the high 16 bits and its binary path in the low bits. A transaction is
routed by interpreting its transaction ID as a little-endian integer and taking
the low `depth` path bits of the active leaf shard.

A leaf shard should split when utilization remains at or above 80 percent for
the configured split window. Sibling leaves should merge when both stay at or
below 20 percent utilization for 24 hours at the target anchor interval.
Shard-tree commitments serialize the sorted leaf set and require a complete,
non-overlapping partition of the root shard.

Each leaf shard has an independent fee floor. The next floor is an integer EWMA
of the previous floor and recent floor-eligible fee rate. At confirmation, the
floor portion of each fee is burned and the remaining tip is paid to the fruit
miner. A transaction with fee below its required burned fee is invalid.

## 8. Relay, Compression, and Data Availability

TensorPoW layers multiple relay optimizations while preserving canonical
consensus bytes. Template/range coding compresses transaction objects. Graphene
uses Bloom filters and IBLT sketches for compact fruit relay. Erlay uses
Minisketch-style set reconciliation for transaction relay. A learned residual
codec may be negotiated only when its frozen weights hash matches the protocol
constant; default consensus payloads remain valid without it. Anchor topology
compression may encode topology as deterministic INT8 low-rank factors only
when reconstruction is bit-exact.

Every compressed object starts with a codec id, uncompressed length, compressed
length, and compressed bytes. Unknown codec ids are invalid for consensus
objects. Decoding either returns exact canonical bytes or fails.

Fruit payloads are also prepared for data availability sampling. Payload bytes
are arranged into a square cell matrix, padded deterministically, and extended
with two-dimensional Reed-Solomon coding over GF(2^8). Light verifiers sample
cells using BLAKE3-derived sample randomness and accept availability only when
enough returned cells carry valid Merkle and Reed-Solomon proofs.

## 9. Issuance and Fees

Genesis has no spendable output, no treasury, and no premine. All TSC enters
circulation through mined fruit and anchor rewards.

The hard cap is 21,000,000 TSC. The first halving epoch contains 10,500,000 TSC
of subsidy over 2,102,400 anchor intervals, corresponding to approximately
2,625,000 TSC per year during the first four years. Later epochs halve the
epoch pool by integer right shift. If a subsidy would exceed the remaining cap,
it is reduced to the remaining unminted supply.

Within an anchor interval, rewards are split by work weight. Each covered fruit
contributes one fruit reward weight. The anchor contributes a work weight
derived from the ratio between the fruit target and active anchor target.
Integer division floors each recipient share, and any remaining matoms are
assigned in ascending recipient-hash order.

Fees are separated into burned floor fees and miner tips. The floor is a
capacity-control mechanism for each shard. Tips compensate the fruit miner that
included the transaction.

## 10. Genesis Commitment

Genesis construction is deterministic and public. The genesis commitment hashes
the mainnet chain id, era marker, whitepaper hash, selected Bitcoin block hash,
selected Ethereum block hash, founder permanent identity public-key hash, empty
UTXO root, initial shard-tree root, and initial fee-floor root:

```text
genesis_commitment = BLAKE3(
    DOMAIN_GENESIS ||
    GENESIS_CHAIN_ID_MAINNET ||
    GENESIS_ERA_MARKER ||
    whitepaper_hash ||
    bitcoin_block_hash ||
    ethereum_block_hash ||
    founder_pubkey_hash ||
    empty_utxo_root ||
    initial_shard_tree_root ||
    initial_fee_floor_root
)
```

The genesis anchor uses the zero parent sentinel, empty fruit set root, empty
parent candidate root, root shard tree, zero fee floor for the root shard, and
the computed genesis commitment. Genesis pays no reward and creates no UTXO.
The genesis anchor identifier is the BLAKE3 hash of the serialized genesis
anchor, so the first fruit's `latest_anchor` reference is chain-separated by the
public genesis inputs; ordinary anchors use their header hash because their
bodies are committed by header roots.

This whitepaper's reproducible BLAKE3 artifact is produced by
`docs/whitepaper/build_whitepaper.py` from the normalized Markdown source. The
resulting `docs/whitepaper/tensorpow.pdf.blake3` file is the whitepaper hash
input consumed by the genesis ceremony tooling; the normalized source hash is
published as a reproducibility check.

## 11. Security and Determinism Evidence

The protocol's security arguments are conditional on canonical validation,
honest majority work assumptions appropriate to proof-of-work systems, and
successful execution of the implementation's regression suites. This paper does
not claim that public-testnet operation, the full adversarial suite, or the
six-hour multi-platform soak has completed. Those are separate launch gates.

Current implementation evidence is tied to these test files:

| Claim area | Evidence files |
|---|---|
| Bit-exact INT8-to-INT32 PoW determinism | `tests/determinism/test_baseline.py`, `tests/determinism/test_pow_kernel.py`, `tests/unit/test_pow_challenge.py`, `tests/unit/test_pow_kernel.py`, `tests/unit/test_pow_verify_miner.py` |
| Header, block, and Merkle canonicalization | `tests/unit/test_headers.py`, `tests/unit/test_blocks.py`, `tests/unit/test_hash.py` |
| Ed25519 dispatch and Bech32m address validation | `tests/unit/test_signatures.py`, `tests/unit/test_address.py` |
| Transaction decoding, scripts, lock checks, and malformed input rejection | `tests/unit/test_transaction.py`, `tests/unit/test_script.py` |
| UTXO roots, inclusion proofs, and non-inclusion proofs | `tests/unit/test_utxo.py`, `tests/unit/test_utxo_sync.py` |
| GHOSTDAG ordering and finality thresholds | `tests/unit/test_ghostdag.py`, `tests/unit/test_finality.py` |
| Shard routing, split/merge rules, fee floors, and mempool conflicts | `tests/unit/test_shard_tree.py`, `tests/unit/test_mempool.py` |
| Relay compression, topology coding, and data availability sampling | `tests/unit/test_template_codec.py`, `tests/unit/test_graphene.py`, `tests/unit/test_erlay.py`, `tests/unit/test_learned_codec.py`, `tests/unit/test_topology_codec.py`, `tests/unit/test_turbine.py`, `tests/unit/test_das.py` |
| Storage atomicity and node/RPC integration boundaries | `tests/unit/test_storage_rocksdb.py`, `tests/unit/test_node.py`, `tests/unit/test_rpc_server.py`, `tests/integration/test_rpc_server.py` |

The missing launch evidence is explicit: adversarial scenario files under
`tests/adversarial/test_*.py` and a six-hour soak driver at `scripts/soak_test.py`
must exist and pass before their corresponding claims can be included in a
genesis-ready claim set.

## 12. Conclusion

TensorPoW combines proof-of-work with a tensor-native work function, a BlockDAG
fruit layer, anchor checkpoints, pure UTXO accounting, hierarchical mempool
sharding, fee-floor burns, and deterministic genesis commitment. Its consensus
surface is intentionally byte-level: if two conforming nodes receive the same
canonical data, they must compute the same hashes, roots, PoW verdicts, and UTXO
state on all supported hardware backends.

The result is a currency design where mining maps directly onto modern INT8
matrix hardware while preserving the conservative validation requirements of a
permissionless monetary network.
