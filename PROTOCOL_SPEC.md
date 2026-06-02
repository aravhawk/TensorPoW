# TensorPoW / Tensorcoin Protocol Specification

**Status:** Authoritative protocol specification.
**Scope:** Mainnet consensus, networking, state, wallet-facing transaction format,
and genesis construction.
**Rule:** Values in the constants table are normative. Other sections reference
constants by name so implementations do not copy magic numbers into consensus code.

This document uses "MUST", "MUST NOT", "SHOULD", and "MAY" in their normal
standards sense. All consensus integers are unsigned and little-endian unless a
field explicitly says otherwise.

---

## 1. Overview & Invariants

TensorPoW is a permissionless proof-of-work BlockDAG chain whose work function is
a deterministic signed INT8 matrix multiplication with INT32 accumulation. The
currency unit is TSC. The ledger model is pure UTXO: there are no accounts, no
global mutable balances, and no shared state outside canonical UTXO transitions.

The protocol has two block classes:

- **Fruits** carry transactions and permanent fruit work.
- **Anchors** commit ordered fruit sets, shard-tree state, and fee-floor state.
  Anchor difficulty is adjusted by WTEMA to target `ANCHOR_INTERVAL_MS`.

Consensus implementations MUST preserve these invariants:

- Same inputs produce bit-identical consensus bytes on CUDA, MPS, x86 CPU, and
  ARM CPU.
- Numeric PoW operations are `POW_INPUT_DTYPE -> POW_ACCUM_DTYPE` and are
  serialized by byte-level canonicalization.
- Fruit target is permanent. WTEMA only adjusts anchor target.
- UTXO conflicts are resolved by BlockDAG ordering, not by account state.
- Genesis has an empty UTXO set. All TSC enters through mining rewards.
- Mempool sharding is a binary tree with depth at most `SHARD_MAX_DEPTH`.
- Protocol changes after genesis require a hardfork on the same chain.

Every consensus hash is computed as `BLAKE3(domain_byte || payload)` unless the
section explicitly specifies BLAKE3 XOF or BLAKE3 keyed mode.

---

## 2. Cryptographic Primitives

### Hashing

The only consensus hash is BLAKE3 with `HASH_LEN_BYTES` output bytes. All hash
values in block headers, transaction IDs, Merkle roots, outpoints, and addresses
MUST be exactly `HASH_LEN_BYTES`.

Domain-separated hashing:

1. The first byte is the domain constant from the constants table.
2. The rest of the payload is the canonical byte encoding defined by the
   relevant section.
3. The output is the first `HASH_LEN_BYTES` of BLAKE3.

BLAKE3 XOF is used only where a section says XOF. XOF readers MUST read exactly
the requested byte count and MUST NOT skip, seek, or platform-adapt the stream.

Keyed BLAKE3 and BLAKE3 key derivation are permitted for wallet and transport
helpers, but consensus serialization MUST NOT depend on secret keys.

### Signatures

`SIG_TYPE_ED25519` is the only active signature type at genesis. Ed25519 public
keys are `ED25519_PUBLIC_KEY_BYTES`, private keys are `ED25519_PRIVATE_KEY_BYTES`
seed bytes, and signatures are `ED25519_SIGNATURE_BYTES`.

Transaction inputs carry a `sig_type` field. Implementations MUST dispatch
signature verification through a table equivalent to:

```text
SIG_DISPATCH = {
    SIG_TYPE_ED25519: ed25519_verify,
}
```

Unknown signature types MUST fail validation. `SIG_TYPE_ML_DSA_RESERVED` is a
reserved identifier and MUST NOT validate before a hardfork activates it.

Signature verification MUST use a constant-time library implementation. Batch
verification MAY be used, but the result MUST be identical to verifying each
signature independently and rejecting if any item fails. Batch verification is a
non-consensus optimization: implementations MUST NOT accept a batch result under
rules that would reject any item when checked independently.

---

## 3. PoW Kernel

The PoW kernel takes two matrices with dimensions
`POW_MATRIX_DIM x POW_MATRIX_DIM`. Each input cell is signed INT8. The output
matrix is signed INT32. The canonical operation is row-major matrix
multiplication:

```text
C[row, col] = sum(A[row, k] * B[k, col]) for k in [0, POW_MATRIX_DIM)
```

The accumulator MUST be at least `POW_ACCUM_BITS` and MUST NOT wrap. The valid
input range is `INT8_MIN_VALUE` through `INT8_MAX_VALUE`. Any non-INT8 input,
wrong shape, non-contiguous interpretation mismatch, NaN, float tensor, or
backend-promoted floating operation is invalid for consensus.

Canonical output bytes are produced by serializing each INT32 cell in row-major
order as `POW_ACCUM_BYTES` little-endian signed twos-complement bytes. The PoW
digest is:

```text
pow_digest = BLAKE3(DOMAIN_POW_OUTPUT || canonical_output_bytes)
```

The target comparison interprets `pow_digest` and the target as unsigned
little-endian integers. A proof passes when:

```text
int_le(pow_digest) <= int_le(target)
```

Targets MUST be exactly `TARGET_BYTES`. Fruit proofs use `FRUIT_TARGET_LE`.
Anchor proofs use the current WTEMA anchor target.

Backends MAY use CUDA, MPS, or CPU, but all MUST return the same canonical bytes.
MPS implementations MUST fall back to CPU if a deterministic INT8 path is not
available. Consensus code MUST set deterministic backend flags where the tensor
library exposes them.

### Challenge Matrix Construction

PoW challenge construction yields two signed INT8 matrices, `A` and `B`. The
matrices are generated from the candidate header and are part of the consensus
proof. Implementations MUST NOT use randomness, floating point, platform RNGs,
or backend-dependent tensor initialization.

Fruit challenge preimage:

```text
fruit_pow_preimage =
    DOMAIN_POW_CHALLENGE_FRUIT ||
    version_u16_le ||
    sig_type_supported_u16_le ||
    parent_count_u16_le ||
    effective_parent_hashes ||
    latest_anchor ||
    tx_merkle_root ||
    timestamp_ms_u64_le ||
    shard_id_u32_le ||
    nonce_u64_le
```

Anchor challenge preimage:

```text
anchor_pow_preimage =
    DOMAIN_POW_CHALLENGE_ANCHOR ||
    version_u16_le ||
    parent_anchor ||
    fruit_set_root ||
    parent_candidate_root ||
    shard_tree_state_root ||
    fee_floor_set_root ||
    anchor_reward_root ||
    timestamp_ms_u64_le ||
    nonce_u64_le
```

`effective_parent_hashes` are encoded in the canonical order defined in section
4. The BLAKE3 XOF inputs are:

```text
A_bytes = BLAKE3_XOF(DOMAIN_POW_MATRIX_A || pow_preimage, POW_MATRIX_BYTES)
B_bytes = BLAKE3_XOF(DOMAIN_POW_MATRIX_B || pow_preimage, POW_MATRIX_BYTES)
```

Each byte maps to signed INT8 by `byte - 256` when the byte is `>= 128`,
otherwise by the byte value unchanged. Bytes fill matrices in row-major order.
The verifier MUST reconstruct both matrices from the decoded header and reject a
proof if either matrix cannot be generated exactly.

---

## 4. Block Formats

All block encodings are length-delimited where a field is variable length. A
decoder MUST reject extra trailing bytes, non-canonical lengths, duplicate set
members, unsorted sets where sorting is required, and any field outside its
validation range.

### Ordered Merkle Roots

`tx_merkle_root`, `fruit_set_root`, `parent_candidate_root`, fee-floor list
roots, and `anchor_reward_root` use the same ordered Merkle construction with
different domain bytes.

For a domain `D` and an ordered item list:

```text
empty_root = BLAKE3(D || MERKLE_EMPTY_TAG || item_count_u32_le)
leaf[i] = BLAKE3(D || MERKLE_LEAF_TAG || i_u32_le || item_len_u32_le || item)
parent[level, i] = BLAKE3(D || MERKLE_NODE_TAG || level_u16_le || i_u32_le || left || right)
```

If a level has an odd number of nodes, the missing right node is:

```text
BLAKE3(D || MERKLE_EMPTY_TAG || level_u16_le || i_u32_le)
```

The root of a single item is its leaf. The root of an empty list is
`empty_root`. Item order is part of the commitment. Duplicate items are invalid
in fruit sets and parent candidate lists.

### Fruit Header

Fruit header hash:

```text
fruit_hash = BLAKE3(DOMAIN_FRUIT_HEADER || fruit_header_bytes)
```

Fruit header byte layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `version` | `uint16` | `U16_BYTES` | MUST equal `FORMAT_EPOCH`. |
| `sig_type_supported` | `uint16` bitmask | `U16_BYTES` | MUST include `SIG_TYPE_ED25519_BIT`. Unknown active bits invalid. |
| `parent_selected` | hash | `HASH_LEN_BYTES` | MUST reference a known fruit or equal `GENESIS_PARENT_HASH` only for genesis. |
| `parent_bitmap_len` | `uint16` | `U16_BYTES` | MUST be `<= PARENT_BITMAP_MAX_BYTES`. |
| `parent_bitmap` | bytes | `parent_bitmap_len` | Additional parents over the latest-anchor candidate window. |
| `latest_anchor` | hash | `HASH_LEN_BYTES` | MUST reference the latest known anchor chosen by the miner. |
| `tx_merkle_root` | hash | `HASH_LEN_BYTES` | MUST equal the Merkle root of the fruit body transactions. |
| `timestamp_ms` | `uint64` | `U64_BYTES` | MUST pass timestamp rules in section 7. |
| `shard_id` | encoded shard id | `U32_BYTES` | MUST pass shard-id validation in section 11. |
| `nonce` | `uint64` | `U64_BYTES` | Any value in `uint64` range. |

`parent_bitmap` uses little-endian bit numbering: bit `i` is
`parent_bitmap[i // BYTE_BITS] & (1 << (i % BYTE_BITS))`. The bitmap is evaluated
against the parent candidate list in the anchor body referenced by
`latest_anchor`. That list is committed by the anchor header's
`parent_candidate_root`. The effective parent set is `parent_selected` plus each
candidate with a set bit. The effective parent hash order for PoW and
serialization-derived checks is `parent_selected`, then selected candidate
hashes in parent-candidate-list order. The set MUST NOT contain duplicates. Bits
beyond the candidate list MUST be zero.

### Fruit Body

Fruit body byte layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `header_len` | `uint16` | `U16_BYTES` | MUST equal encoded fruit header length. |
| `header` | bytes | `header_len` | MUST decode as a valid fruit header. |
| `tx_count` | `uint16` | `U16_BYTES` | MUST be at least `MIN_FRUIT_TX_COUNT`. |
| `transactions` | repeated `len || tx` | variable | Each length is `uint16`; aggregate payload `<= MAX_FRUIT_PAYLOAD_BYTES`. |

The first transaction MUST be a coinbase transaction for the fruit miner. All
other transactions MUST be valid non-coinbase transactions assigned to the
fruit's `shard_id`. `tx_merkle_root` is the Merkle root of transaction IDs in
body order. Empty transaction bodies are invalid because the coinbase is
mandatory.

### Anchor Header

Anchor header hash:

```text
anchor_hash = BLAKE3(DOMAIN_ANCHOR_HEADER || anchor_header_bytes)
```

Anchor header byte layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `version` | `uint16` | `U16_BYTES` | MUST equal `FORMAT_EPOCH`. |
| `parent_anchor` | hash | `HASH_LEN_BYTES` | MUST reference a known anchor or `GENESIS_PARENT_HASH` only for genesis. |
| `fruit_set_root` | hash | `HASH_LEN_BYTES` | MUST equal root of covered fruit hashes. |
| `parent_candidate_root` | hash | `HASH_LEN_BYTES` | MUST equal root of the parent candidate list. |
| `shard_tree_state_root` | hash | `HASH_LEN_BYTES` | MUST equal serialized shard tree commitment. |
| `fee_floor_set_root` | hash | `HASH_LEN_BYTES` | MUST equal serialized fee-floor commitment. |
| `anchor_reward_root` | hash | `HASH_LEN_BYTES` | MUST equal root of anchor reward outputs. |
| `timestamp_ms` | `uint64` | `U64_BYTES` | MUST pass timestamp rules in section 7. |
| `nonce` | `uint64` | `U64_BYTES` | Any value in `uint64` range. |

### Anchor Body

Anchor body byte layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `header_len` | `uint16` | `U16_BYTES` | MUST equal encoded anchor header length. |
| `header` | bytes | `header_len` | MUST decode as a valid anchor header. |
| `covered_fruit_count` | `uint32` | `U32_BYTES` | MUST be nonzero after genesis. |
| `covered_fruit_hashes` | sorted hashes | count times `HASH_LEN_BYTES` | Strict ascending byte order, no duplicates. |
| `parent_candidate_count` | `uint32` | `U32_BYTES` | MUST be `<= PARENT_CANDIDATE_MAX_COUNT`. |
| `parent_candidate_hashes` | ordered hashes | count times `HASH_LEN_BYTES` | Canonical parent candidate order, no duplicates. |
| `shard_tree_len` | `uint32` | `U32_BYTES` | MUST be `<= SHARD_TREE_MAX_BYTES`. |
| `shard_tree_bytes` | bytes | `shard_tree_len` | MUST decode to a valid shard tree. |
| `fee_floor_count` | `uint32` | `U32_BYTES` | MUST match leaf shard count. |
| `fee_floor_entries` | repeated | count times fixed entry | Sorted by `shard_id`. |
| `anchor_reward_output_count` | `uint16` | `U16_BYTES` | MUST be zero in genesis. |
| `anchor_reward_outputs` | repeated `len || output` | variable | Each output length is `uint16`; root MUST match `anchor_reward_root`. |
| `genesis_commitment` | hash | `HASH_LEN_BYTES` | MUST be zero except in the genesis anchor. |

Each fee-floor entry is:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `shard_id` | encoded shard id | `U32_BYTES` | Valid leaf shard id. |
| `floor_matoms_per_kb` | `uint64` | `U64_BYTES` | Any value in `uint64` range. |

`parent_candidate_hashes` are the selected frontier a miner may additionally
reference in subsequent fruit headers. The list is sorted by descending blue
score, then ascending fruit hash, and truncated to `PARENT_CANDIDATE_MAX_COUNT`.

The fee-floor set root is the ordered Merkle root of canonical fee-floor entries
using `DOMAIN_FEE_FLOOR`. The shard tree root is
`BLAKE3(DOMAIN_SHARD_TREE || shard_tree_bytes)`.
The anchor reward root is the ordered Merkle root of canonical transaction-output
encodings using `DOMAIN_ANCHOR_REWARD_ROOT`.

---

## 5. BlockDAG and GHOSTDAG Ordering

The fruit graph is a directed acyclic graph. Each fruit references its selected
parent and optional additional parents. Parent edges always point to earlier
fruits by timestamp and known DAG reachability. A fruit that would introduce a
cycle is invalid.

For this protocol version, TensorPoW uses a fixed GHOSTDAG parameter:

```text
k = DYNAMIC_K_MIN
```

All consensus paths that classify, order, or select fruit DAG candidates MUST
use this fixed value. The dynamic formula below is reserved for a future
versioned activation and MUST NOT be used for consensus until the observation
inputs are specified as deterministic chain-derived values:

```text
reserved_dynamic_k = clamp(
    ceil(DYNAMIC_K_FACTOR * observed_lambda * observed_d_max_ms / MS_PER_SECOND
         * ln(1 / delta)),
    DYNAMIC_K_MIN,
    DYNAMIC_K_MAX,
)
```

`observed_lambda` is the observed fruit rate in fruits per second over
`DYNAMIC_K_OBSERVATION_ANCHORS`. `observed_d_max_ms` is the network propagation
delay estimate bounded by `DYNAMIC_K_D_MAX_MIN_MS` and
`DYNAMIC_K_D_MAX_MAX_MS`. `delta` is `DYNAMIC_K_DELTA_NUM /
DYNAMIC_K_DELTA_DEN`. These values are advisory/reserved in this version; an
implementation that feeds local wall-clock or network measurements into
consensus K is non-conformant.

GHOSTDAG classification follows the greedy algorithm from PHANTOM/GHOSTDAG:

1. Select the parent with the highest accumulated blue work; break ties by
   lexicographic fruit hash.
2. Build the candidate blue set from selected-parent blues plus merge-set
   candidates.
3. A candidate is blue if its anticone size against the candidate blue set is
   `<= k`.
4. Candidates that fail are red.
5. Topological order sorts by blue score, then timestamp, then fruit hash.

Implementations MUST cache blue score and selected parent metadata, but cached
values are only valid if recomputation from canonical parents yields the same
bytes. An attacker with less than `ADVERSARY_REORG_COMPUTE_PCT_LIMIT` of total
work MUST NOT be able to reorg a transaction past `FINALITY_ECONOMIC_BLUE_DEPTH`
in the adversarial regression suite.

---

## 6. Fruit and Anchor Work

Each fruit proof uses `FRUIT_TARGET_LE`. Fruit target never changes. Its target
is derived from `FRUIT_WORK_OPS`, `POW_OPS_PER_MATMUL`, and the full
`TARGET_BITS` hash space:

```text
FRUIT_TARGET = floor((2^TARGET_BITS - 1) * POW_OPS_PER_MATMUL / FRUIT_WORK_OPS)
```

The anchor genesis target is:

```text
ANCHOR_INITIAL_TARGET = floor(FRUIT_TARGET / ANCHOR_WORK_MULTIPLIER)
```

Anchor work is adjusted only by WTEMA. A valid anchor proof MUST satisfy the
current anchor target computed from prior anchors. A valid fruit proof MUST
satisfy the permanent fruit target regardless of network hashrate.

Total reward weight inside an anchor interval is work proportional:

- Each covered fruit contributes `FRUIT_REWARD_WEIGHT`.
- The anchor contributes `anchor_work_weight`, computed from the ratio between
  `FRUIT_TARGET_LE` and the active anchor target and clamped to `uint64`.

The interval subsidy is distributed by integer floor division over those weights.
Remainder subunits are assigned by ascending recipient hash, one subunit at a
time, until exhausted.

---

## 7. Timing

Consensus timestamps are Unix time in milliseconds. A block timestamp MUST be
greater than the median timestamp of the previous `MEDIAN_TIME_PAST_WINDOW`
blocks of the same class and MUST NOT be more than `MAX_FUTURE_DRIFT_MS` ahead
of the receiving node's adjusted network time.

Anchor WTEMA:

1. Use at most `WTEMA_WINDOW_ANCHORS` most recent parent-chain anchor intervals.
   If fewer intervals exist, use all available intervals. The window is
   recomputed from the supplied parent-chain history for each target calculation;
   implementations MUST NOT carry WTEMA accumulator state forward outside this
   window.
2. Clamp each observed interval to
   `[ANCHOR_INTERVAL_MS / WTEMA_MAX_ADJUSTMENT_FACTOR,
   ANCHOR_INTERVAL_MS * WTEMA_MAX_ADJUSTMENT_FACTOR]`.
3. Compute the exponentially weighted interval ratio in fixed-point integer
   arithmetic with `WTEMA_ALPHA_NUM / WTEMA_ALPHA_DEN`: initialize
   `ratio_fp = 2^64`, then fold the clamped intervals in chronological order as
   `sample_fp = interval_ms * 2^64 / ANCHOR_INTERVAL_MS` and
   `ratio_fp = ((WTEMA_ALPHA_DEN - WTEMA_ALPHA_NUM) * ratio_fp
   + WTEMA_ALPHA_NUM * sample_fp) / WTEMA_ALPHA_DEN`, using integer floor
   division at each `/`.
4. Multiply previous target by that ratio.
5. Clamp per-anchor target movement to the same max-adjustment factor.
6. Clamp final target to `[ANCHOR_MIN_TARGET_LE, ANCHOR_MAX_TARGET_LE]`.

Finality tiers:

| Tier | Requirement |
|---|---|
| `Seen` | Transaction appears in any valid fruit known to the node. |
| `Fast` | Containing fruit has blue depth `FINALITY_FAST_BLUE_DEPTH`. |
| `Economic` | Containing fruit has blue depth `FINALITY_ECONOMIC_BLUE_DEPTH`. |
| `Settlement` | Containing fruit has blue depth `FINALITY_SETTLEMENT_BLUE_DEPTH` and is covered by `FINALITY_SETTLEMENT_ANCHOR_DEPTH` anchors. |
| `AnchorSecured` | Containing fruit is committed by at least `FINALITY_ANCHOR_SECURED_DEPTH` anchors. |

Wall-clock estimates are advisory only and MUST NOT affect consensus validity.

---

## 8. UTXO State

An outpoint is:

| Field | Type | Bytes |
|---|---:|---:|
| `tx_id` | hash | `HASH_LEN_BYTES` |
| `output_index` | `uint32` | `U32_BYTES` |

Outpoint bytes are `tx_id || output_index_le`. The outpoint key is:

```text
outpoint_key = BLAKE3(DOMAIN_OUTPOINT || outpoint_bytes)
```

A UTXO serializes as:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `outpoint` | bytes | `OUTPOINT_BYTES` | MUST match key. |
| `amount_matoms` | `uint64` | `U64_BYTES` | MUST be nonzero and within supply cap. |
| `template_id` | `uint16` | `U16_BYTES` | Known output template. |
| `owner_pubkey_hash` | hash | `HASH_LEN_BYTES` | BLAKE3 pubkey hash or template-defined owner hash. |
| `locktime_ms` | `uint64` | `U64_BYTES` | Zero means unlocked by wall time. |
| `lockheight` | `uint64` | `U64_BYTES` | Zero means unlocked by DAG height. |
| `payload_len` | `uint16` | `U16_BYTES` | MUST be `<= TX_OUTPUT_PAYLOAD_MAX_BYTES`. |
| `payload` | bytes | `payload_len` | Template-specific data. |

The UTXO root is a compact sparse Merkle tree keyed by `outpoint_key`. Leaves
are:

```text
leaf = BLAKE3(DOMAIN_MERKLE_LEAF || outpoint_key || BLAKE3(DOMAIN_UTXO || utxo_bytes))
```

Internal nodes are:

```text
node = BLAKE3(DOMAIN_MERKLE_NODE || depth_u16_le || left_hash || right_hash)
```

Empty subtrees use deterministic empty hashes:

```text
empty_hash(depth) = BLAKE3(DOMAIN_MERKLE_EMPTY || depth_u16_le)
```

Inclusion and non-inclusion proofs MUST verify against the root using only
canonical node encodings. A proof with duplicate siblings, wrong depth, or
non-canonical side markers is invalid.

---

## 9. Address Format

A Tensorcoin address is a lowercase Bech32m string with HRP `ADDRESS_HRP`.

Address payload:

```text
pubkey_hash = BLAKE3(DOMAIN_ADDRESS || ed25519_public_key)
address = bech32m_encode(ADDRESS_HRP, convertbits(pubkey_hash, 8, 5, pad=True))
```

Validation rules:

- Entire string MUST be lowercase.
- HRP MUST equal `ADDRESS_HRP`.
- Separator MUST be present.
- Encoding MUST be Bech32m with checksum constant `BECH32M_CONST`.
- Decoded payload MUST convert back to exactly `ADDRESS_HASH_BYTES`.
- Mixed case, wrong HRP, invalid checksum, non-canonical padding, or invalid
  character set MUST fail.

Implementations SHOULD report a likely typo position when the checksum library
can identify one, but error-location reporting is not consensus critical.

---

## 10. Transaction Format

Transaction ID:

```text
tx_id = BLAKE3(DOMAIN_TX_ID || canonical_tx_bytes)
```

Signature hash:

```text
sighash = BLAKE3(DOMAIN_TX_SIGHASH || canonical_tx_bytes_with_empty_witnesses || input_index_le)
```

Transaction byte layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `version` | `uint16` | `U16_BYTES` | MUST equal `FORMAT_EPOCH`. |
| `sig_type` | `uint8` | `U8_BYTES` | MUST equal an active signature type. |
| `locktime_ms` | `uint64` | `U64_BYTES` | Wall-time lock; zero disabled. |
| `lockheight` | `uint64` | `U64_BYTES` | DAG-height lock; zero disabled. |
| `input_count` | `uint16` | `U16_BYTES` | Coinbase has zero; non-coinbase MUST be nonzero. |
| `inputs` | repeated input | variable | Canonical order as provided by signer. |
| `output_count` | `uint16` | `U16_BYTES` | MUST be nonzero. |
| `outputs` | repeated output | variable | Output index is list position. |

Serialized transaction size MUST be `<= MAX_TX_BYTES`.

Input layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `previous_outpoint` | outpoint | `OUTPOINT_BYTES` | MUST exist and be unspent for non-coinbase. |
| `sequence` | `uint32` | `U32_BYTES` | Reserved for relative locks; currently `TX_SEQUENCE_FINAL`. |
| `witness_len` | `uint16` | `U16_BYTES` | MUST be `<= TX_WITNESS_MAX_BYTES`. |
| `witness` | bytes | `witness_len` | Template-specific stack encoding. |

Output layout:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `amount_matoms` | `uint64` | `U64_BYTES` | Nonzero except explicitly allowed burn output. |
| `template_id` | `uint16` | `U16_BYTES` | Known template. |
| `locktime_ms` | `uint64` | `U64_BYTES` | Zero disabled. |
| `lockheight` | `uint64` | `U64_BYTES` | Zero disabled. |
| `payload_len` | `uint16` | `U16_BYTES` | MUST be `<= TX_OUTPUT_PAYLOAD_MAX_BYTES`. |
| `payload` | bytes | `payload_len` | Template-specific bytes. |

Coinbase transactions MUST have `input_count == COINBASE_INPUT_COUNT`, MUST be
the first transaction in a fruit, and MUST NOT mint more than the fruit's
assigned subsidy plus tips.

### Script Templates

Scripts are stack machines with byte opcodes from the constants table. Script
execution MUST be deterministic, bounded by `SCRIPT_MAX_OPS`, and fail on stack
underflow, invalid opcode, invalid signature type, non-minimal push, or exceeding
`SCRIPT_MAX_STACK_ITEMS`.

Active output templates:

- `TEMPLATE_PKH`: payload is `owner_pubkey_hash`. Witness is
  `signature || public_key`; validation checks BLAKE3 public-key hash and
  `OP_CHECKSIG`.
- `TEMPLATE_MULTISIG`: payload is `threshold || pubkey_count || pubkeys`.
  `pubkey_count` MUST be `<= MULTISIG_MAX_KEYS`, and threshold MUST be in
  `[1, pubkey_count]`.
- `TEMPLATE_HASHLOCK`: payload is `hash || inner_template_payload`. Witness
  MUST reveal a preimage whose BLAKE3 hash matches, then satisfy the inner
  template.

All fees are:

```text
fee = sum(inputs) - sum(outputs)
```

Transactions with negative fees are invalid. Transactions below the current
shard fee floor are not relayable and not includable unless they are coinbase.

---

## 11. Hierarchical Sharding

A shard id is a `uint32`:

```text
shard_id = (depth << SHARD_ID_DEPTH_SHIFT) | path
```

`depth` MUST be in `[0, SHARD_MAX_DEPTH]`. `path` MUST be less than
`2^depth`; unused high path bits MUST be zero. The root shard is depth zero and
path zero.

Routing:

```text
tx_route_int = int_le(tx_id)
path = tx_route_int & ((1 << depth) - 1)
```

A transaction belongs to the unique leaf shard whose encoded path matches the
low `depth` bits of `tx_route_int`.

Shard utilization for a leaf over a window is:

```text
payload_bytes_confirmed / (MAX_FRUIT_PAYLOAD_BYTES * fruit_slots_observed)
```

Split rule: a leaf SHOULD split when utilization is at least
`SHARD_SPLIT_THRESHOLD_PCT` for `SHARD_SPLIT_WINDOW_ANCHORS` consecutive anchors.
A split is invalid if it would exceed `SHARD_MAX_DEPTH`.

Merge rule: sibling leaves SHOULD merge when both siblings stay at or below
`SHARD_MERGE_THRESHOLD_PCT` for `SHARD_MERGE_WINDOW_ANCHORS`. Concurrent
split/merge operations for different parents may be committed in one anchor.
Conflicting operations for the same parent are ordered by anchor order; later
conflicts are queued.

Shard-tree commitments use:

```text
shard_tree_root = BLAKE3(DOMAIN_SHARD_TREE || canonical_shard_tree_bytes)
```

`canonical_shard_tree_bytes` are:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `leaf_count` | `uint32` | `U32_BYTES` | MUST be nonzero. |
| `leaf_shard_ids` | repeated `uint32` | `leaf_count * U32_BYTES` | Strict ascending by encoded shard id. |

The leaf set MUST describe a complete non-overlapping binary partition of the
root shard. Any gap, overlap, duplicate, invalid depth, or leaf past
`SHARD_MAX_DEPTH` is invalid.

---

## 12. Mempool Relay

Each leaf shard has an independent mempool and fee floor. A node MUST validate a
transaction before admitting or relaying it:

1. Canonical decode succeeds.
2. Size is `<= MAX_TX_BYTES`.
3. Shard route matches the destination shard.
4. Inputs are available or accepted as missing-input orphans within local
   non-consensus limits.
5. Fee rate is at least the shard fee floor.
6. Scripts pass against the node's current UTXO view.

Fee floor calculation per shard:

```text
recent_rate = floor(total_floor_eligible_fees * BYTES_PER_KB / max(1, total_payload_bytes))
next_floor = floor((FEE_FLOOR_EWMA_PREV_WEIGHT * previous_floor
                   + FEE_FLOOR_EWMA_NEW_WEIGHT * recent_rate)
                  / FEE_FLOOR_EWMA_DEN)
```

The window is the shard's last `FEE_FLOOR_WINDOW_FRUITS` confirmed fruits. Empty
windows use `recent_rate = FEE_FLOOR_MIN_MATOMS_PER_KB`.

At confirmation:

```text
burned_fee = floor(fee_floor_matoms_per_kb * tx_size_bytes / BYTES_PER_KB)
tip = fee - burned_fee
```

`burned_fee` is permanently destroyed. `tip` goes to the miner of the fruit that
included the transaction. A transaction with `fee < burned_fee` is invalid.

Fruit selection MUST respect `MAX_FRUIT_PAYLOAD_BYTES`, prioritize by fee rate,
and break ties by `tx_id` ascending.

---

## 13. Networking

The wire protocol uses libp2p with:

- TCP and QUIC transports.
- Noise security.
- yamux multiplexing.
- Kademlia peer discovery.
- Gossipsub 2.0 topic gossip.

Consensus topic names:

| Payload | Topic |
|---|---|
| Fruits | `TOPIC_FRUITS` |
| Anchors | `TOPIC_ANCHORS` |
| Shard transactions | `TOPIC_TXS_PREFIX || shard_id_hex || TOPIC_TXS_SUFFIX` |

Messages MUST begin with:

| Field | Type | Bytes | Validation |
|---|---:|---:|---|
| `magic` | bytes | `WIRE_MAGIC_BYTES` | MUST equal `WIRE_MAGIC`. |
| `message_type` | `uint16` | `U16_BYTES` | Known message type. |
| `payload_len` | `uint32` | `U32_BYTES` | MUST be `<= WIRE_MAX_PAYLOAD_BYTES`. |
| `payload` | bytes | `payload_len` | Type-specific canonical payload. |
| `checksum` | bytes | `WIRE_CHECKSUM_BYTES` | First checksum bytes of BLAKE3 payload hash. |

Message type registry:

| Message | Value | Payload |
|---|---:|---|
| `MSG_TYPE_FRUIT` | `0x0001` | Canonical fruit bytes. |
| `MSG_TYPE_ANCHOR` | `0x0002` | Canonical anchor bytes. |
| `MSG_TYPE_TX` | `0x0003` | Canonical transaction bytes. |
| `MSG_TYPE_GRAPHENE_SKETCH` | `0x0004` | Graphene sketch bytes. |
| `MSG_TYPE_ERLAY_SKETCH` | `0x0005` | Erlay reconciliation sketch bytes. |
| `MSG_TYPE_DAS_REQUEST` | `0x0006` | DAS sample request bytes. |
| `MSG_TYPE_DAS_RESPONSE` | `0x0007` | DAS sample response bytes. |

Malformed messages MUST be dropped. Repeated malformed messages SHOULD reduce
peer score and can disconnect the peer. No networking message is trusted until
its consensus payload validates.

The protocol does not enforce a minimum bandwidth. Operators SHOULD provision
`RECOMMENDED_BANDWIDTH_BYTES_PER_SEC` or more for reliable public routing.

---

## 14. Compression Stack

Compression is layered and must be bit-exact:

1. Template transaction codec strips fields implied by `template_id` and default
   signature type.
2. Deterministic integer range coding compresses residual bytes.
3. Graphene compact fruit relay uses Bloom filters plus IBLT sketches.
4. Erlay transaction reconciliation uses Minisketch-style set reconciliation
   every `ERLAY_INTERVAL_MS`.
5. Learned residual codec MAY be negotiated only when the frozen weights hash
   matches `LEARNED_CODEC_WEIGHTS_HASH`; default consensus payloads remain valid
   without it.
6. Anchor topology codec MAY encode topology as deterministic INT8 low-rank
   factors only when reconstruction is bit-exact.

`CODEC_TEMPLATE_RANGE` transaction objects use the common compression object
header (`codec_id_u16_le || uncompressed_len_u32_le ||
compressed_len_u32_le`) followed by:

`TEMPLATE_CODEC_MAGIC` (`TPTC`) || `TEMPLATE_RANGE_CODER_ADAPTIVE` (`0x01`)
|| `template_payload_len_u32_le` || `range_bitstream`.

The template payload strips the default transaction version, default
`SIG_TYPE_ED25519`, final input sequences, standard witness lengths, zero
locks, and signer PKH repetitions. The range bitstream is a deterministic
32-bit integer arithmetic/range coder over 256 byte symbols with initial
frequency `1` for every symbol and adaptive integer rescaling. It uses no
floating point. Decoders MUST range-decode the template payload, reconstruct the
canonical transaction, then recompress and byte-compare the complete
`CODEC_TEMPLATE_RANGE` object.

Learned transaction codec objects wrap the template-range transaction object,
not raw transaction bytes. `CODEC_LEARNED` compressed bytes are
`LEARNED_CODEC_MAGIC` (`TPLC`) || `template_object_hash` ||
`template_object_len_varint` || `residual_count_varint` || repeated
`skip_varint || actual_byte`. The INT8 prediction vector is loaded from
`LEARNED_CODEC_WEIGHTS_PATH`; each prediction byte is
`int8_weight + LEARNED_CODEC_INT8_ZERO_POINT`. Residual indexes are strictly
ascending through zero-skips, the template object hash MUST match, and decoders
MUST reject non-canonical residuals.

Anchor topology raw bytes are
`parent_candidate_count_u32_le || parent_candidate_hashes`. `CODEC_TOPOLOGY`
compressed bytes are `TOPOLOGY_CODEC_MAGIC` (`TPTF`) ||
`TOPOLOGY_AFFINE_INT8` (`0x01`) || `parent_candidate_count_u32_le` ||
`raw_topology_hash` || `base_int8[HASH_LEN_BYTES]` ||
`row_delta_int8[HASH_LEN_BYTES]`. Row `i`, column `j` reconstructs as
`base[j] + i * row_delta[j] mod 256`; the raw hash MUST match the canonical raw
bytes. Encoders MUST use `CODEC_RAW` with the canonical raw bytes unless the
topology factor object is smaller, and decoders MUST reject non-canonical
fallbacks or factors.

Every compressed object begins with `codec_id` as a `uint16` followed by
`uncompressed_len_u32_le`, `compressed_len_u32_le`, and compressed bytes.
Unknown codec IDs are invalid for consensus objects and unsupported for relay
objects. Decoding MUST either return the exact canonical bytes or fail; lossy
decoding is never valid.

Codec registry:

| Codec | Value | Meaning |
|---|---:|---|
| `CODEC_RAW` | `0x0000` | No compression. |
| `CODEC_TEMPLATE_RANGE` | `0x0001` | Template codec plus deterministic range coder. |
| `CODEC_GRAPHENE` | `0x0002` | Graphene relay sketch. |
| `CODEC_ERLAY` | `0x0003` | Erlay reconciliation sketch. |
| `CODEC_LEARNED` | `0x0004` | Frozen learned residual codec. |
| `CODEC_TOPOLOGY` | `0x0005` | Anchor topology codec. |

Graphene relay MUST fall back to full fruit request when reconstruction fails.
Erlay sketches MUST reject malformed lengths and invalid field encodings before
set reconciliation.

---

## 15. Data Availability Sampling

Fruit payloads are arranged into a square data matrix, padded with deterministic
zero chunks, and extended by two-dimensional Reed-Solomon coding.

Rules:

- Cell size is `DAS_CELL_BYTES`.
- If the data matrix side is `k`, the extended matrix side is
  `DAS_RS_EXTENSION_FACTOR * k`. Reed-Solomon coding is over GF(2^8) with
  primitive polynomial `DAS_RS_PRIMITIVE_POLY`, generator
  `DAS_RS_GENERATOR`, first consecutive root
  `DAS_RS_FIRST_CONSECUTIVE_ROOT`, and field exponent
  `DAS_RS_FIELD_EXPONENT`. Encoders extend every data row from `k` to `2k`
  symbols, then extend every resulting column from `k` to `2k` symbols.
- A light verifier samples `DAS_SAMPLES_PER_FRUIT` uniformly using
  `BLAKE3(DOMAIN_DAS_SAMPLE || fruit_hash || sample_index_le)` as randomness.
- A fruit is considered available to a light verifier when at least
  `DAS_SAMPLE_SUCCESS_THRESHOLD_PCT` of requested samples are returned with
  valid Merkle and Reed-Solomon proofs.
- A peer that serves invalid cells is penalized. A peer that cannot serve cells
  is not trusted for availability.

DAS sample request payloads are fixed-width:

`fruit_hash[32] || sample_index_u32_le || row_u32_le || column_u32_le`.

DAS sample response payloads carry the requested proof:

`row_u32_le || column_u32_le || row_count_u32_le || repeated(row_cell[256] ||
merkle_proof) || column_count_u32_le || repeated(column_cell[256] ||
merkle_proof)`.

Each Merkle proof is
`leaf_index_u32_le || leaf_count_u32_le || sibling_count_u32_le ||
repeated(is_left_u8 || sibling_hash[32])`, where `is_left_u8` is exactly `0` or
`1`. Row and column witness counts MUST match, be nonzero, and fit the
Reed-Solomon extended side limit.

The DAS security regression suite MUST show detection probability greater than
`DAS_WITHHOLDING_DETECTION_PCT` when an attacker withholds
`DAS_WITHHOLDING_PCT` of cells after `DAS_SAMPLES_PER_FRUIT` samples.

---

## 16. Economics

The atomic unit is the matom. `MATOMS_PER_TSC` matoms equal one TSC.

The supply cap is `MAX_SUPPLY_TSC`. Genesis has no spendable outputs and no
allocation. The first spendable TSC comes from mined fruit and anchor rewards.

Halving epochs are measured in anchors:

```text
HALVING_INTERVAL_ANCHORS = HALVING_YEARS * DAYS_PER_YEAR * HOURS_PER_DAY
                           * MINUTES_PER_HOUR
```

The first epoch subsidy pool is `INITIAL_EPOCH_SUBSIDY_TSC`. Each later epoch
uses integer half of the prior epoch pool. Per-anchor interval subsidy is:

```text
epoch_pool = INITIAL_EPOCH_SUBSIDY_MATOMS >> epoch_index
base = epoch_pool // HALVING_INTERVAL_ANCHORS
remainder = epoch_pool % HALVING_INTERVAL_ANCHORS
subsidy = base + (1 if anchor_index_in_epoch < remainder else 0)
```

If total minted supply would exceed `MAX_SUPPLY_MATOMS`, subsidy is reduced to
the remaining unminted supply. If no supply remains, subsidy is zero. Fees still
burn and tips still pay miners.

The reward split within each anchor interval is pure work proportional as
defined in section 6. Fruit miners claim the fruit pool plus realized tips
through fruit coinbase transactions. Anchor miners claim at most the anchor pool
through `anchor_reward_outputs`, which are committed by `anchor_reward_root` in
the anchor header and use anchor-hash-derived outpoints. Coinbase outputs mature
after `COINBASE_MATURITY_ANCHORS`. Transactions covered by an anchor are
validated against that anchor's post-application height for lockheight and
coinbase-maturity checks. Therefore a fruit coinbase output created in anchor
height `H` is first spendable by a transaction covered in anchor height
`H + COINBASE_MATURITY_ANCHORS`.

---

## 17. Genesis Block Construction

Genesis construction is deterministic and public.

Inputs:

- `GENESIS_CHAIN_ID_MAINNET`.
- `GENESIS_ERA_MARKER`.
- Whitepaper PDF BLAKE3 hash.
- Recent Bitcoin block hash selected by `GENESIS_BTC_SELECTION_RULE`.
- Recent Ethereum block hash selected by `GENESIS_ETH_SELECTION_RULE`.
- Founder permanent identity pubkey hash.
- Empty UTXO root.
- Initial shard-tree root.
- Initial fee-floor root.

Selection rules:

- The Bitcoin block is the latest Bitcoin block before ceremony start UTC with
  at least `GENESIS_BTC_CONFIRMATIONS` confirmations.
- The Ethereum block is the latest finalized Ethereum block before ceremony
  start UTC.

Genesis commitment:

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

The genesis anchor uses `GENESIS_PARENT_HASH` as `parent_anchor`, empty fruit
set root, empty parent candidate root, root shard tree, zero fee floor for the
root shard, and the computed `genesis_commitment` field in its body. Genesis
creates no UTXO and pays no reward.

The genesis anchor hash is `BLAKE3(serialized_genesis_anchor)` so the chain ID,
era marker, whitepaper hash, external block hashes, founder key hash, and empty
state roots are bound into the anchor identifier used by the first fruit and by
the first ordinary anchor. Ordinary post-genesis anchors use the anchor-header
hash as their block hash because their bodies are already committed by header
roots.

---

## 18. Hard-Coded Constants Table

### Encoding and Sizes

| Constant | Value | Meaning |
|---|---:|---|
| `FORMAT_EPOCH` | `0` | Canonical serialization identifier, not a protocol generation. |
| `ENDIANNESS` | `little` | Integer byte order. |
| `U8_BYTES` | `1` | `uint8` byte width. |
| `U16_BYTES` | `2` | `uint16` byte width. |
| `U32_BYTES` | `4` | `uint32` byte width. |
| `U64_BYTES` | `8` | `uint64` byte width. |
| `BYTE_BITS` | `8` | Bits per byte. |
| `HASH_LEN_BYTES` | `32` | BLAKE3 digest length used by consensus. |
| `TARGET_BITS` | `256` | Target comparison bit width. |
| `TARGET_BYTES` | `32` | Target byte width. |
| `GENESIS_PARENT_HASH` | `00` repeated `32` bytes | Parent hash sentinel for genesis only. |

### Domain Bytes

| Constant | Value | Meaning |
|---|---:|---|
| `DOMAIN_POW_CHALLENGE_FRUIT` | `0x00` | Fruit challenge XOF domain. |
| `DOMAIN_POW_CHALLENGE_ANCHOR` | `0x01` | Anchor challenge XOF domain. |
| `DOMAIN_POW_OUTPUT` | `0x02` | PoW output digest domain. |
| `DOMAIN_POW_MATRIX_A` | `0x03` | First PoW matrix XOF domain. |
| `DOMAIN_POW_MATRIX_B` | `0x04` | Second PoW matrix XOF domain. |
| `DOMAIN_FRUIT_HEADER` | `0x10` | Fruit header hash domain. |
| `DOMAIN_ANCHOR_HEADER` | `0x11` | Anchor header hash domain. |
| `DOMAIN_TX_MERKLE_ROOT` | `0x12` | Transaction Merkle root domain. |
| `DOMAIN_FRUIT_SET_ROOT` | `0x13` | Anchor covered-fruit root domain. |
| `DOMAIN_PARENT_CANDIDATE_ROOT` | `0x14` | Parent candidate root domain. |
| `DOMAIN_ANCHOR_REWARD_ROOT` | `0x15` | Anchor reward output root domain. |
| `DOMAIN_TX_ID` | `0x20` | Transaction ID domain. |
| `DOMAIN_TX_SIGHASH` | `0x21` | Transaction signature hash domain. |
| `DOMAIN_OUTPOINT` | `0x22` | Outpoint key domain. |
| `DOMAIN_UTXO` | `0x23` | UTXO value hash domain. |
| `DOMAIN_MERKLE_LEAF` | `0x30` | Merkle leaf domain. |
| `DOMAIN_MERKLE_NODE` | `0x31` | Merkle internal node domain. |
| `DOMAIN_MERKLE_EMPTY` | `0x32` | Empty subtree domain. |
| `MERKLE_LEAF_TAG` | `0x00` | Ordered Merkle leaf tag. |
| `MERKLE_NODE_TAG` | `0x01` | Ordered Merkle internal node tag. |
| `MERKLE_EMPTY_TAG` | `0x02` | Ordered Merkle empty node tag. |
| `DOMAIN_ADDRESS` | `0x40` | Public-key hash address domain. |
| `DOMAIN_GENESIS` | `0x50` | Genesis commitment domain. |
| `DOMAIN_SHARD_TREE` | `0x60` | Shard-tree commitment domain. |
| `DOMAIN_FEE_FLOOR` | `0x61` | Fee-floor commitment domain. |
| `DOMAIN_DAS_SAMPLE` | `0x70` | DAS sample randomness domain. |

### Cryptography and Addresses

| Constant | Value | Meaning |
|---|---:|---|
| `SIG_TYPE_ED25519` | `0` | Active Ed25519 signature type. |
| `SIG_TYPE_ML_DSA_RESERVED` | `1` | Reserved post-quantum signature type. |
| `SIG_TYPE_ED25519_BIT` | `1 << SIG_TYPE_ED25519` | Fruit header capability bit. |
| `ED25519_PUBLIC_KEY_BYTES` | `32` | Ed25519 public key size. |
| `ED25519_PRIVATE_KEY_BYTES` | `32` | Ed25519 seed size. |
| `ED25519_SIGNATURE_BYTES` | `64` | Ed25519 signature size. |
| `ADDRESS_HRP` | `tsc` | Bech32m human-readable part. |
| `ADDRESS_HASH_BYTES` | `32` | Address payload after 5-bit decode. |
| `BECH32M_CONST` | `0x2bc830a3` | Bech32m checksum constant. |

### PoW and Difficulty

| Constant | Value | Meaning |
|---|---:|---|
| `POW_MATRIX_DIM` | `1024` | Matrix rows and columns. |
| `POW_INPUT_DTYPE` | `int8` | Input matrix cell type. |
| `POW_ACCUM_DTYPE` | `int32` | Output accumulator type. |
| `POW_ACCUM_BITS` | `32` | Accumulator bit width. |
| `POW_ACCUM_BYTES` | `4` | Serialized accumulator width. |
| `POW_MATRIX_INPUTS` | `2` | Number of generated PoW input matrices. |
| `POW_MATRIX_BYTES` | `1048576` | Bytes per INT8 matrix. |
| `INT8_MIN_VALUE` | `-128` | Minimum signed INT8. |
| `INT8_MAX_VALUE` | `127` | Maximum signed INT8. |
| `POW_OPS_PER_MAC` | `2` | Multiply and add counted as two operations. |
| `POW_OPS_PER_MATMUL` | `2147483648` | `POW_OPS_PER_MAC * POW_MATRIX_DIM^3`. |
| `FRUIT_WORK_OPS` | `6000000000` | Permanent fruit work target in operations. |
| `ANCHOR_WORK_MULTIPLIER` | `1000` | Initial anchor work multiple over fruit work. |
| `FRUIT_TARGET_LE` | `195719183ec6044d312767ab3e988ffe757563505b5c45760f1923cf803fa05b` | Permanent fruit target bytes. |
| `ANCHOR_INITIAL_TARGET_LE` | `c4e77a534b84e06c453fe6097f9953a30b5d8b9e968aa9cc16383daccc741700` | Initial anchor target bytes. |
| `ANCHOR_MIN_TARGET_LE` | `0100000000000000000000000000000000000000000000000000000000000000` | Hardest allowed anchor target. |
| `ANCHOR_MAX_TARGET_LE` | `FRUIT_TARGET_LE` | Easiest allowed anchor target. |

### Timing, DAA, and Finality

| Constant | Value | Meaning |
|---|---:|---|
| `MS_PER_SECOND` | `1000` | Milliseconds per second. |
| `ANCHOR_INTERVAL_MS` | `60000` | Anchor wall-clock target. |
| `MEDIAN_TIME_PAST_WINDOW` | `11` | Timestamp median window. |
| `MAX_FUTURE_DRIFT_MS` | `120000` | Max accepted future timestamp drift. |
| `WTEMA_WINDOW_ANCHORS` | `100` | Anchor intervals used for WTEMA samples. |
| `WTEMA_ALPHA_NUM` | `1` | WTEMA alpha numerator. |
| `WTEMA_ALPHA_DEN` | `8` | WTEMA alpha denominator. |
| `WTEMA_MAX_ADJUSTMENT_FACTOR` | `4` | Max per-anchor adjustment factor. |
| `FINALITY_FAST_BLUE_DEPTH` | `5` | Fast finality threshold. |
| `FINALITY_ECONOMIC_BLUE_DEPTH` | `20` | Economic finality threshold. |
| `FINALITY_SETTLEMENT_BLUE_DEPTH` | `100` | Settlement blue-depth threshold. |
| `FINALITY_ANCHOR_SECURED_DEPTH` | `1` | Anchor-secured threshold. |
| `FINALITY_SETTLEMENT_ANCHOR_DEPTH` | `6` | Settlement anchor-depth threshold. |

### GHOSTDAG

| Constant | Value | Meaning |
|---|---:|---|
| `DYNAMIC_K_FACTOR` | `2` | Reserved leading factor in inactive dynamic K formula. |
| `DYNAMIC_K_MIN` | `15` | Consensus GHOSTDAG K for this version. |
| `DYNAMIC_K_MAX` | `10000` | Reserved maximum K for inactive dynamic formula. |
| `DYNAMIC_K_DELTA_NUM` | `1` | Reserved delta numerator. |
| `DYNAMIC_K_DELTA_DEN` | `1000000` | Reserved delta denominator. |
| `DYNAMIC_K_OBSERVATION_ANCHORS` | `100` | Reserved observation window for lambda. |
| `DYNAMIC_K_D_MAX_MIN_MS` | `100` | Reserved lower propagation-delay bound. |
| `DYNAMIC_K_D_MAX_MAX_MS` | `5000` | Reserved upper propagation-delay bound. |
| `PARENT_BITMAP_MAX_BYTES` | `1250` | Bitmap bytes for `DYNAMIC_K_MAX` candidates. |
| `PARENT_CANDIDATE_MAX_COUNT` | `10000` | Maximum anchor parent candidates. |
| `ADVERSARY_REORG_COMPUTE_PCT_LIMIT` | `40` | Regression-test attacker limit. |

### Transactions and Scripts

| Constant | Value | Meaning |
|---|---:|---|
| `MAX_TX_BYTES` | `8192` | Maximum serialized transaction size. |
| `MAX_FRUIT_PAYLOAD_BYTES` | `8192` | Maximum serialized non-header fruit payload. |
| `MIN_FRUIT_TX_COUNT` | `1` | Coinbase is mandatory. |
| `OUTPOINT_BYTES` | `36` | Transaction hash plus output index. |
| `TX_SEQUENCE_FINAL` | `0xffffffff` | Final sequence value. |
| `TX_WITNESS_MAX_BYTES` | `2048` | Max input witness bytes. |
| `TX_OUTPUT_PAYLOAD_MAX_BYTES` | `2048` | Max output payload bytes. |
| `COINBASE_INPUT_COUNT` | `0` | Coinbase transaction input count. |
| `SCRIPT_MAX_BYTES` | `1024` | Max script bytes. |
| `SCRIPT_MAX_OPS` | `256` | Max script op count. |
| `SCRIPT_MAX_STACK_ITEMS` | `1024` | Max script stack items. |
| `SCRIPT_MAX_ELEMENT_BYTES` | `520` | Max pushed element bytes. |
| `MULTISIG_MAX_KEYS` | `15` | Max public keys in multisig. |
| `TEMPLATE_PKH` | `0` | Pay-to-public-key-hash template. |
| `TEMPLATE_MULTISIG` | `1` | N-of-M multisig template. |
| `TEMPLATE_HASHLOCK` | `2` | Hashlock template. |
| `OP_PUSH` | `0x01` | Push length-prefixed bytes. |
| `OP_DUP` | `0x76` | Duplicate stack top. |
| `OP_HASH256` | `0xaa` | BLAKE3 hash to `HASH_LEN_BYTES`. |
| `OP_CHECKSIG` | `0xac` | Ed25519 signature check. |
| `OP_CHECKMULTISIG` | `0xae` | Multisig verification. |
| `OP_HASHLOCK` | `0xb1` | Hash preimage check. |
| `OP_CHECKLOCKTIME` | `0xb2` | Wall-time lock check. |
| `OP_CHECKLOCKHEIGHT` | `0xb3` | DAG-height lock check. |
| `OP_VERIFY` | `0x69` | Require truthy stack top. |

### Sharding and Mempool

| Constant | Value | Meaning |
|---|---:|---|
| `SHARD_MAX_DEPTH` | `16` | Maximum binary shard-tree depth. |
| `SHARD_ID_DEPTH_SHIFT` | `16` | Shift for encoded shard depth. |
| `SHARD_SPLIT_THRESHOLD_PCT` | `80` | Split utilization threshold. |
| `SHARD_SPLIT_WINDOW_ANCHORS` | `10` | Split sustained window. |
| `SHARD_MERGE_THRESHOLD_PCT` | `20` | Merge utilization threshold per sibling. |
| `SHARD_MERGE_WINDOW_ANCHORS` | `1440` | Merge window at target anchor interval. |
| `SHARD_TREE_MAX_BYTES` | `262144` | Maximum serialized shard tree bytes. |
| `BYTES_PER_KB` | `1000` | Fee-rate denominator. |
| `FEE_FLOOR_WINDOW_FRUITS` | `1024` | Fee-floor recent fruit window. |
| `FEE_FLOOR_MIN_MATOMS_PER_KB` | `0` | Minimum fee floor. |
| `FEE_FLOOR_EWMA_PREV_WEIGHT` | `7` | Previous floor EWMA weight. |
| `FEE_FLOOR_EWMA_NEW_WEIGHT` | `1` | Recent rate EWMA weight. |
| `FEE_FLOOR_EWMA_DEN` | `8` | Fee-floor EWMA denominator. |

### Networking, Relay, and DAS

| Constant | Value | Meaning |
|---|---:|---|
| `WIRE_MAGIC` | `54504f57` | ASCII `TPOW`. |
| `WIRE_MAGIC_BYTES` | `4` | Wire magic length. |
| `WIRE_CHECKSUM_BYTES` | `4` | Payload checksum length. |
| `WIRE_MAX_PAYLOAD_BYTES` | `16777216` | Max wire payload size. |
| `MSG_TYPE_FRUIT` | `0x0001` | Fruit wire message. |
| `MSG_TYPE_ANCHOR` | `0x0002` | Anchor wire message. |
| `MSG_TYPE_TX` | `0x0003` | Transaction wire message. |
| `MSG_TYPE_GRAPHENE_SKETCH` | `0x0004` | Graphene sketch wire message. |
| `MSG_TYPE_ERLAY_SKETCH` | `0x0005` | Erlay sketch wire message. |
| `MSG_TYPE_DAS_REQUEST` | `0x0006` | DAS request wire message. |
| `MSG_TYPE_DAS_RESPONSE` | `0x0007` | DAS response wire message. |
| `CODEC_ID_BYTES` | `2` | Compression codec ID width. |
| `CODEC_RAW` | `0x0000` | Raw payload codec. |
| `CODEC_TEMPLATE_RANGE` | `0x0001` | Template/range transaction codec. |
| `TEMPLATE_CODEC_MAGIC` | `TPTC` | Template codec body magic. |
| `TEMPLATE_RANGE_CODER_ADAPTIVE` | `0x01` | Adaptive integer range coder. |
| `CODEC_GRAPHENE` | `0x0002` | Graphene sketch codec. |
| `CODEC_ERLAY` | `0x0003` | Erlay sketch codec. |
| `CODEC_LEARNED` | `0x0004` | Learned residual codec. |
| `CODEC_TOPOLOGY` | `0x0005` | Anchor topology codec. |
| `TOPIC_FRUITS` | `tensorpow/fruits/main` | Fruit gossip topic. |
| `TOPIC_ANCHORS` | `tensorpow/anchors/main` | Anchor gossip topic. |
| `TOPIC_TXS_PREFIX` | `tensorpow/txs/` | Shard tx topic prefix. |
| `TOPIC_TXS_SUFFIX` | `/main` | Shard tx topic suffix. |
| `RECOMMENDED_BANDWIDTH_BYTES_PER_SEC` | `10000000` | Operator bandwidth recommendation. |
| `ERLAY_INTERVAL_MS` | `8000` | Erlay reconciliation interval. |
| `GRAPHENE_RECEIVER_MEMPOOL_PCT` | `99` | Compression acceptance scenario. |
| `GRAPHENE_TARGET_COMPRESSION_PCT` | `95` | Expected compact-block compression. |
| `GRAPHENE_BLOOM_BITS_PER_TX` | `2` | Graphene Bloom filter sizing factor. |
| `GRAPHENE_BLOOM_HASH_COUNT` | `3` | Graphene Bloom hash count. |
| `GRAPHENE_IBLT_HASH_COUNT` | `3` | Graphene IBLT cell positions per key. |
| `GRAPHENE_IBLT_KEY_BYTES` | `8` | Graphene IBLT relay key width. |
| `GRAPHENE_IBLT_MIN_CELLS` | `3` | Minimum Graphene IBLT cell count. |
| `TURBINE_SIM_NODE_COUNT` | `1000` | Relay simulation node count. |
| `TURBINE_MAX_PROPAGATION_MS` | `200` | Relay simulation latency target. |
| `TURBINE_DROPOUT_PCT` | `30` | Relay dropout resilience target. |
| `TURBINE_DATA_SHARDS` | `16` | Reed-Solomon data shards per fruit payload. |
| `TURBINE_PARITY_SHARDS` | `8` | Reed-Solomon parity shards per fruit payload. |
| `TURBINE_MIN_FANOUT` | `2` | Minimum adaptive relay fanout. |
| `TURBINE_MAX_FANOUT` | `16` | Maximum adaptive relay fanout. |
| `TURBINE_CHUNK_REPLICATION` | `8` | Deterministic chunk custodian replication factor. |
| `DAS_CELL_BYTES` | `256` | DAS matrix cell size. |
| `DAS_SAMPLE_REQUEST_BYTES` | `44` | Fixed DAS sample request payload size. |
| `DAS_RS_EXTENSION_FACTOR` | `2` | Reed-Solomon side expansion factor. |
| `DAS_RS_PRIMITIVE_POLY` | `0x11d` | Reed-Solomon GF(2^8) primitive polynomial. |
| `DAS_RS_GENERATOR` | `2` | Reed-Solomon GF(2^8) generator. |
| `DAS_RS_FIRST_CONSECUTIVE_ROOT` | `0` | Reed-Solomon first consecutive root. |
| `DAS_RS_FIELD_EXPONENT` | `8` | Reed-Solomon GF field exponent. |
| `DAS_SAMPLE_SUCCESS_THRESHOLD_PCT` | `75` | Availability success threshold. |
| `DAS_SAMPLES_PER_FRUIT` | `10` | Samples per fruit for light verifier. |
| `DAS_CONFIDENCE_PCT` | `99` | Claimed light-verifier confidence. |
| `DAS_WITHHOLDING_PCT` | `50` | Withholding attack test level. |
| `DAS_WITHHOLDING_DETECTION_PCT` | `99` | Required detection probability. |
| `LEARNED_CODEC_WEIGHTS_PATH` | `data/learned_codec.npz` | Frozen learned codec weights path. |
| `LEARNED_CODEC_WEIGHTS_HASH` | `GENESIS_PARENT_HASH` | All-zero hash disables learned codec until frozen weights exist. |
| `LEARNED_CODEC_EXTRA_COMPRESSION_PCT` | `15` | Learned codec compression target. |
| `LEARNED_CODEC_INT8_ZERO_POINT` | `128` | Converts signed INT8 predictions into bytes. |
| `TOPOLOGY_CODEC_COMPRESSION_PCT` | `20` | Anchor topology codec target. |

### Storage and Economics

| Constant | Value | Meaning |
|---|---:|---|
| `STORAGE_BACKEND` | `RocksDB` | Production storage backend. |
| `ROCKSDB_CF_HEADERS` | `headers` | Header column family. |
| `ROCKSDB_CF_BODIES` | `bodies` | Body column family. |
| `ROCKSDB_CF_UTXO` | `utxo` | UTXO column family. |
| `ROCKSDB_CF_DAG` | `dag` | DAG metadata column family. |
| `ROCKSDB_CF_SHARD_TREE` | `shard_tree` | Shard-tree column family. |
| `ROCKSDB_CF_FEE_FLOORS` | `fee_floors` | Fee-floor column family. |
| `ROCKSDB_CF_MEMPOOL` | `mempool` | Mempool column family. |
| `ROCKSDB_WRITE_TARGET_PER_SEC` | `10000` | Storage benchmark target. |
| `FRUIT_REWARD_WEIGHT` | `1` | Reward weight per covered fruit. |
| `MATOMS_PER_TSC` | `100000000` | Atomic subunits per TSC. |
| `MAX_SUPPLY_TSC` | `21000000` | Hard supply cap in TSC. |
| `MAX_SUPPLY_MATOMS` | `2100000000000000` | Hard supply cap in matoms. |
| `HALVING_YEARS` | `4` | Halving period. |
| `DAYS_PER_YEAR` | `365` | Subsidy schedule year length. |
| `HOURS_PER_DAY` | `24` | Subsidy schedule day length. |
| `MINUTES_PER_HOUR` | `60` | Subsidy schedule hour length. |
| `HALVING_INTERVAL_ANCHORS` | `2102400` | Anchors per halving epoch. |
| `INITIAL_ISSUANCE_TSC_PER_YEAR` | `2625000` | Approximate first-year issuance. |
| `INITIAL_EPOCH_SUBSIDY_TSC` | `10500000` | First halving epoch subsidy pool. |
| `INITIAL_EPOCH_SUBSIDY_MATOMS` | `1050000000000000` | First epoch subsidy in matoms. |
| `COINBASE_MATURITY_ANCHORS` | `100` | Coinbase spend maturity. |

### Genesis and Public Network

| Constant | Value | Meaning |
|---|---:|---|
| `GENESIS_CHAIN_ID_MAINNET` | `tensorpow-mainnet` | Mainnet chain identifier. |
| `GENESIS_CHAIN_ID_TESTNET` | `tensorpow-testnet` | Testnet chain identifier. |
| `GENESIS_ERA_MARKER` | `tensorpow-2026` | Genesis era marker. |
| `GENESIS_BTC_CONFIRMATIONS` | `6` | Bitcoin confirmation requirement. |
| `GENESIS_BTC_SELECTION_RULE` | latest pre-ceremony block with required confirmations | Bitcoin anchor selection. |
| `GENESIS_ETH_SELECTION_RULE` | latest pre-ceremony finalized block | Ethereum anchor selection. |
| `PUBLIC_TESTNET_MIN_DAYS` | `30` | Public testnet minimum duration. |
| `PUBLIC_TESTNET_MIN_NODES` | `100` | Public testnet participation target. |
| `SOAK_TEST_DURATION_HOURS` | `6` | Determinism stress duration. |

---

## References

- BLAKE3 specification: https://github.com/BLAKE3-team/BLAKE3-specs
- Ed25519 / EdDSA: https://www.rfc-editor.org/rfc/rfc8032
- Bech32 and Bech32m: https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki
  and https://github.com/bitcoin/bips/blob/master/bip-0350.mediawiki
- PHANTOM/GHOSTDAG: https://eprint.iacr.org/2018/104.pdf
