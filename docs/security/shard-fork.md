# Shard-Fork Regression

Attack model: concurrent anchors try to apply duplicate shard splits, duplicate
merges, and independent split operations in the same shard-tree update window.

Regression: `tests/adversarial/test_shard_fork.py` applies same-parent
conflicts and independent operations through the canonical shard-tree updater,
then round-trips the resulting commitment state.

Expected result: first same-parent operations apply, later conflicts are queued,
independent operations apply, and malformed overlapping partitions are rejected.
