# Spam Fee-Floor Regression

Attack model: after congestion raises a shard fee floor, an attacker floods the
network with zero-fee transactions while subsequent empty or zero-fee fruits try
to drag the floor down.

Regression: `tests/adversarial/test_spam.py` warms the floor with deterministic
confirmed full-fruit samples, applies empty-fruit samples, then attempts a
zero-fee transaction flood.

Expected result: the floor remains positive and every non-coinbase zero-fee spam
transaction is rejected with `below_fee_floor`.
