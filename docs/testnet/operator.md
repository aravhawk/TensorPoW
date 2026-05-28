# TensorPoW Public Testnet Operator Guide

TensorPoW testnet uses chain id `tensorpow-testnet`. It exists to satisfy the
pre-mainnet gate: at least 30 continuous days, at least 100 unique nodes, and
public attempts of the adversarial scenarios documented under `docs/security/`.

## Bootstrap Node

After building `docs/testnet/testnet-genesis.json`, generate a bootstrap
configuration:

```bash
/opt/anaconda3/bin/python scripts/testnet_ops.py bootstrap-config \
  --data-dir ~/tensorpow/data/testnet-bootstrap \
  --genesis docs/testnet/testnet-genesis.json \
  --listen-host 0.0.0.0 \
  --public-host bootstrap.tensorpow.org \
  --p2p-port 28333 \
  --rpc-host 127.0.0.1 \
  --rpc-port 28332 \
  --out docs/testnet/bootstrap-config.json
```

The generated TOML enables networking, binds the node to the expected
`tensorpow-testnet` genesis hash, and records the public P2P multiaddress from
`--public-host`, not the wildcard listen address.
RPC should remain loopback-only unless an operator places it behind explicit
authentication and rate limiting.

## Testnet Genesis

Build the public testnet genesis artifact with the same ceremony inputs as the
mainnet rehearsal, but with `tensorpow-testnet` as the chain id:

```bash
/opt/anaconda3/bin/python scripts/testnet_ops.py genesis \
  --whitepaper-hash "$(awk '{print $3}' docs/whitepaper/tensorpow.pdf.blake3)" \
  --bitcoin-block-hash "$BITCOIN_BLOCK_HASH" \
  --ethereum-block-hash "$ETHEREUM_BLOCK_HASH" \
  --founder-pubkey-hash "$FOUNDER_PUBKEY_HASH" \
  --out docs/testnet/testnet-genesis.json
```

The artifact includes a `public_testnet_gate` object. That gate is not satisfied
until external monitoring records the required duration and node count.

## Faucet

The faucet helper builds ordinary signed transactions from a funded faucet
wallet. The helper does not mint coins and does not bypass mempool validation:

```bash
/opt/anaconda3/bin/python scripts/testnet_ops.py faucet-tx \
  --wallet ~/tensorpow/data/testnet-faucet.json \
  --password "$FAUCET_PASSWORD" \
  --utxos ~/tensorpow/data/testnet-faucet-utxos.json \
  --to "$RECIPIENT_TSC_ADDRESS" \
  --amount 100000000 \
  --fee 1000
```

Submit the returned `rawtx` to `sendrawtx` on a synced node.

## Minimal Explorer

Run the local explorer/faucet HTTP facade against a node RPC endpoint:

```bash
export TENSORPOW_TESTNET_FAUCET_PASSWORD='use-a-hardware-backed-secret'
/opt/anaconda3/bin/python scripts/testnet_ops.py serve \
  --host 127.0.0.1 \
  --port 28380 \
  --rpc-url http://127.0.0.1:28332/rpc \
  --faucet-wallet ~/tensorpow/data/testnet-faucet.json \
  --faucet-utxos ~/tensorpow/data/testnet-faucet-utxos.json \
  --faucet-amount 100000000 \
  --faucet-fee 1000
```

Endpoints:

- `GET /health`
- `GET /explorer/mempool`
- `GET /explorer/block?hash=<64-hex-block-hash>`
- `GET /explorer/tx?txid=<64-hex-transaction-id>`
- `GET /explorer/finality?hash=<64-hex-block-hash>`
- `GET /explorer/utxos?address=tsc...`
- `POST /faucet` with `{"address":"tsc..."}` to build, sign, relay, and return
  a faucet transaction when the server was started with faucet wallet options.
  Without those options the endpoint returns readiness metadata for operators.
- `POST /faucet` with `{"rawtx":"..."}` to relay a prepared faucet transaction.

## Launch Gate Evidence

Before mainnet genesis, publish one JSON evidence file and validate it locally:

```bash
/opt/anaconda3/bin/python scripts/launch_gates.py testnet \
  --genesis docs/testnet/testnet-genesis.json \
  --evidence docs/testnet/public-testnet-evidence.json
```

The evidence object must contain:

- `testnet_genesis_hash`, matching the supplied `tensorpow-testnet` genesis
  artifact.
- `bootstrap_multiaddrs`, with at least one public bootstrap address. IP
  literals must be globally routable; loopback, private, and link-local
  addresses are rejected.
- `faucet_url` and `explorer_url`, published over HTTPS on globally routable
  public hosts; loopback, private IPs, `.local`, and single-label hosts are
  rejected.
- `start_time_ms` and `end_time_ms`, proving at least 30 continuous days.
- `unique_nodes`, with at least 100 distinct participating node identifiers.
  Each identifier must be the canonical libp2p Ed25519 PeerId announced by the
  node, not an operator-chosen label.
- `node_observations`, one signed monitor observation per unique node. Each
  observation must cover the whole testnet window, meet the minimum uptime
  threshold, include an HTTPS evidence URL, and verify under one of at least
  three independent Ed25519 monitor public keys.
- `monitor_logs`, signed by at least three independent Ed25519 monitor public
  keys, with HTTPS log URLs, BLAKE3 content hashes, the final head anchor hash,
  the final state root, start/final anchor heights, start/final head timestamps,
  total fruits/anchors, zero split count, maximum checkpoint gap no larger than
  one day, and at least one checkpoint hash per public-testnet day. All monitor
  logs must agree on the same final head, state root, chain progress counters,
  and checkpoint hash sequence.
- `consensus_splits`, which must be an explicit empty array.
- `attack_scenarios`, with `attempted: true` and `succeeded: false` for
  `double-spend`, `selfish-mining`, `eclipse`, `long-range`,
  `spam-fee-floor`, and `shard-fork`. Each scenario must also include at least
  one HTTPS `evidence_urls` entry, report BLAKE3 hash, start/end timestamps
  inside the testnet window, and an Ed25519 monitor signature over the full
  scenario report payload.

Monitor observations, monitor logs, attack reports, source-chain observations,
and publication attestations can be signed with:

```bash
export TENSORPOW_MONITOR_KEYSTORE_PASSWORD='use-a-hardware-backed-secret'
/opt/anaconda3/bin/python scripts/launch_gates.py sign-evidence \
  --domain testnet-log \
  --payload docs/testnet/monitor-log-payload.json \
  --wallet ~/tensorpow/data/monitor-keystore.json \
  --out docs/testnet/monitor-log-signed.json
```

Also publish the testnet genesis JSON, bootstrap multiaddresses, faucet
transaction logs, explorer uptime logs, and the raw public records behind every
attack-scenario result.
