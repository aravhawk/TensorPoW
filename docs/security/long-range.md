# Long-Range History Regression

Attack model: after an honest fruit reaches economic finality, an attacker
publishes an alternate branch from genesis with less than 40 percent post-genesis
fruit work.

Regression: `tests/adversarial/test_long_range.py` proves the alternate branch
does not contain the protected fruit, remains red after merge, and cannot replace
the honest selected parent.

Expected result: the protected fruit remains `Economic` after the alternate
history is revealed.
