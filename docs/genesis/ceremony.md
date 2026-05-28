# TensorPoW Genesis Ceremony Runbook

Mainnet genesis is a public ceremony, not a dry-run artifact. Run it only after
the public-testnet gate in `docs/testnet/operator.md` is satisfied.

## Required Inputs

- `GENESIS_CHAIN_ID_MAINNET`: `tensorpow-mainnet`
- `GENESIS_ERA_MARKER`: `tensorpow-2026`
- Whitepaper PDF hash: `docs/whitepaper/tensorpow.pdf.blake3`
- Whitepaper normalized-source hash: `docs/whitepaper/tensorpow.blake3`
- Bitcoin block hash: latest Bitcoin block before ceremony start UTC with at
  least 6 confirmations
- Ethereum block hash: latest finalized Ethereum block before ceremony start UTC
- Public HTTPS evidence URLs for the selected Bitcoin and Ethereum blocks
- Bitcoin confirmation-tip height/hash evidence proving the selected block is
  exactly the latest eligible six-confirmation pre-ceremony block
- Ethereum finalized-head number/hash evidence proving the selected block is the
  latest finalized pre-ceremony block
- BLAKE3 hashes of the exact public source-chain evidence documents and at
  least two independent signed monitor observations for each source chain
- Founder permanent identity public key, generated and sealed in an encrypted
  TensorPoW keystore

## Founder Key

Create or reopen the founder keystore through the ceremony script. The password
is supplied by environment variable so it is not written into shell history:

```bash
export TENSORPOW_FOUNDER_KEYSTORE_PASSWORD='use-a-hardware-backed-secret'
/opt/anaconda3/bin/python scripts/genesis_ceremony.py \
  --whitepaper-hash "$(awk '{print $3}' docs/whitepaper/tensorpow.pdf.blake3)" \
  --ceremony-start-ms "$CEREMONY_START_MS" \
  --bitcoin-block-hash "$BITCOIN_BLOCK_HASH" \
  --bitcoin-block-height "$BITCOIN_BLOCK_HEIGHT" \
  --bitcoin-confirmations "$BITCOIN_CONFIRMATIONS" \
  --bitcoin-confirmation-tip-height "$BITCOIN_CONFIRMATION_TIP_HEIGHT" \
  --bitcoin-confirmation-tip-hash "$BITCOIN_CONFIRMATION_TIP_HASH" \
  --bitcoin-confirmation-tip-observed-at-ms "$BITCOIN_CONFIRMATION_TIP_OBSERVED_AT_MS" \
  --bitcoin-observed-at-ms "$BITCOIN_OBSERVED_AT_MS" \
  --bitcoin-source-content-blake3 "$BITCOIN_SOURCE_CONTENT_BLAKE3" \
  --bitcoin-source-url "$BITCOIN_SOURCE_URL" \
  --ethereum-block-hash "$ETHEREUM_BLOCK_HASH" \
  --ethereum-block-number "$ETHEREUM_BLOCK_NUMBER" \
  --ethereum-finalized-head-number "$ETHEREUM_FINALIZED_HEAD_NUMBER" \
  --ethereum-finalized-head-hash "$ETHEREUM_FINALIZED_HEAD_HASH" \
  --ethereum-finalized-at-ms "$ETHEREUM_FINALIZED_AT_MS" \
  --ethereum-source-content-blake3 "$ETHEREUM_SOURCE_CONTENT_BLAKE3" \
  --ethereum-source-url "$ETHEREUM_SOURCE_URL" \
  --founder-keystore ~/tensorpow/data/founder-mainnet-keystore.json \
  --out docs/genesis/mainnet-genesis.json
```

The script records the founder public key, founder Tensorcoin address,
commitment inputs, serialized genesis anchor, full anchor hash, consensus header
identifier used by descendant fruits/anchors, source-chain selection evidence,
and publication metadata. `BITCOIN_CONFIRMATIONS` must equal `6`,
`BITCOIN_CONFIRMATION_TIP_HEIGHT` must be exactly
`BITCOIN_BLOCK_HEIGHT + BITCOIN_CONFIRMATIONS - 1`,
`ETHEREUM_FINALIZED_HEAD_NUMBER` and `ETHEREUM_FINALIZED_HEAD_HASH` must match
the selected Ethereum block, all observed/finalized timestamps must be no later
than `CEREMONY_START_MS`, and both source URLs must be public HTTPS URLs with a
path.

Before final launch-gate validation, add at least two signed
`source_observations` entries to both source-chain records. Each observation is
signed over the exact selection payload with `scripts/launch_gates.py
sign-evidence --domain source-selection`.

## Publication

Publish the exact `mainnet-genesis.json` and `anchor_hex` to:

- GitHub Release titled `vX.Y.Z`
- IPFS
- Arweave
- Operator mirrors

The release title must follow the `vX.Y.Z` format. The JSON document is the
canonical ceremony record; mirror sites should not rewrite or pretty-print it.

## First Mining

Immediately after publication, start mining against the published genesis
anchor. The launch gate is satisfied only when the first fruit is mined within
one minute of publication and all public hashes match deterministic
recomputation from the ceremony inputs.

## Validation

Anyone can verify the artifact with:

```python
import json
from tensorpow.genesis import artifact_from_json

with open("docs/genesis/mainnet-genesis.json", encoding="utf-8") as handle:
    artifact = artifact_from_json(json.load(handle))

print(artifact.block_hash.hex())
print(artifact.header_hash.hex())
```

After publication and the first mined fruit, validate the launch gate evidence:

```bash
/opt/anaconda3/bin/python scripts/launch_gates.py genesis-publication \
  --ceremony docs/genesis/mainnet-genesis.json \
  --evidence docs/genesis/publication-evidence.json \
  --testnet-genesis docs/testnet/testnet-genesis.json \
  --testnet-evidence docs/testnet/public-testnet-evidence.json \
  --expected-whitepaper-hash "$(awk '{print $3}' docs/whitepaper/tensorpow.pdf.blake3)"
```

The publication evidence must bind the GitHub Release title, GitHub URL, IPFS
CID, Arweave id, mirror URLs, exact ceremony JSON BLAKE3 hash, published anchor
hash and bytes, first fruit bytes, first fruit hash, publication timestamp,
first-fruit timestamp, and at least two independent signed publication
attestations for every GitHub/IPFS/Arweave/mirror target. The GitHub URL must be
the `aravhawk/TensorPoW` release path for the same `vX.Y.Z` tag. The ceremony
document must also bind the selected Bitcoin and Ethereum source-chain records
to the committed hashes, ceremony start time, confirmation/finality evidence,
selection rules, public HTTPS evidence URLs, public source-content hashes, and
independent signed source observations. The attestations must bind each target
to the exact ceremony JSON BLAKE3 hash, genesis anchor hash, publication
timestamp, first-fruit timestamp, and first-fruit hash under at least two
independent Ed25519 monitor public keys per target. The first fruit must
reference the chain-separated genesis anchor identifier, carry the same
consensus timestamp as `first_fruit_at_ms`, and be mined no later than one
minute after publication. The verifier also rejects non-mainnet ceremony
artifacts, checks the expected whitepaper PDF hash, checks that the
public-testnet gate is already satisfied, checks that the first fruit satisfies
`FRUIT_TARGET_LE`, and checks that the public founder key record hashes to the
founder pubkey hash committed in the genesis inputs.
IPFS evidence must be a CIDv0-style or base32 CID string; Arweave evidence must
be a 43-character base64url transaction id.
