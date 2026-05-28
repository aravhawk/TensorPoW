# Double-Spend Regression

Attack model: two fruits carry transactions that spend the same UTXO and are
published as concurrent DAG siblings.

Regression: `tests/adversarial/test_double_spend.py` orders the siblings through
GHOSTDAG, applies them to a node in canonical order, and proves only the first
spend updates UTXO state. The later spend fails with `missing_input`.

Expected result: a conflicting spend cannot create two live outputs or resurrect
the spent outpoint.
