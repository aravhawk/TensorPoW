# TensorPoW

TensorPoW is the reference implementation of a GPU-native proof-of-work
blockchain protocol for Tensorcoin. The protocol is built around deterministic
INT8 tensor computation and a fruit/anchor block structure designed for
cross-backend consensus verification.

The implementation is written in Python 3.12+, with PyTorch for portable tensor
execution, Cython for hot paths, BLAKE3 for hashing, Ed25519 for signatures,
libp2p for peer networking, and RocksDB for durable node state.

## Architecture

- `src/tensorpow/crypto`: hashing, signatures, and address primitives.
- `src/tensorpow/pow`: deterministic tensor challenge generation, verification,
  and mining search.
- `src/tensorpow/chain`: fruit and anchor block formats.
- `src/tensorpow/consensus`: BlockDAG ordering, anchor difficulty adjustment,
  dynamic K, and finality.
- `src/tensorpow/state`: UTXO storage and Merkle commitments.
- `src/tensorpow/tx`: transaction serialization, validation, and scripts.
- `src/tensorpow/mempool`: hierarchical sharding and fee-floor policy.
- `src/tensorpow/net`: libp2p gossip, relay, and data-availability paths.
- `src/tensorpow/storage`: RocksDB-backed persistence.
- `src/tensorpow/node`, `src/tensorpow/miner`, `src/tensorpow/wallet`,
  `src/tensorpow/rpc`, and `src/tensorpow/cli`: full-node orchestration and
  operator-facing tools.

The protocol is specified in `PROTOCOL_SPEC.md`.

## Release Readiness

The reference tree includes unit, integration, adversarial, determinism, lint,
format, type-check, and bounded cross-backend smoke coverage. Before any
irreversible mainnet funds are placed at risk, operators must complete the
external release gates in `docs/testnet/operator.md`,
`docs/genesis/ceremony.md`, and `docs/operations/mac-mini-handoff.md`: a
six-hour CPU/MPS/CPU/CUDA soak with matching consensus reports, a public
30-day/100-node testnet, public genesis publication, and first-fruit evidence.

## Operator Tools

- `tensorpow-mine`: deterministic mining smoke tool.
- `tensorpow-node`: node init/start/stop/status/peers CLI.
- `tensorpow-wallet`: encrypted keystore, address, balance, send, import, and
  export CLI.
- `scripts/soak_test.py`: deterministic multi-platform state soak runner.
- `scripts/mac_mini_handoff.sh`: Mac mini to CUDA-PC verification and soak
  runner.
- `scripts/testnet_ops.py`: testnet genesis, bootstrap config, faucet
  transaction, and explorer/faucet facade helpers.
- `scripts/genesis_ceremony.py`: deterministic genesis ceremony artifact
  builder.

Whitepaper artifacts live under `docs/whitepaper/`; the reproducible normalized
source hash and PDF hash are committed for ceremony input verification.
