# Selfish-Mining Regression

Attack model: an attacker with less than 40 percent post-genesis fruit work
withholds a private branch and publishes it after an honest transaction reaches
economic blue depth.

Regression: `tests/adversarial/test_selfish_mining.py` builds the withheld
branch deterministically, merges it with the honest tip, and checks that
GHOSTDAG keeps the honest tip as selected parent while the attacker branch is
red.

Expected result: the protected honest fruit remains `Economic` and the withheld
branch cannot reorg past economic depth.
