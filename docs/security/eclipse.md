# Eclipse-Isolation Regression

Attack model: an isolated victim receives only adversarial wire payloads,
malformed transactions, and an anchor with invalid shard-tree bytes before it
reconnects to an honest peer.

Regression: `tests/adversarial/test_eclipse.py` validates checksum rejection,
transaction decode rejection, and `bad_shard_tree` rejection, then syncs the
victim from an honest node.

Expected result: invalid eclipse data is not persisted, and honest sync restores
the canonical fruit and anchor state.
