# Exact game-theory foundations

Requested by Trent during the next training-development increment. Status:
**numeric corpus and independent oracle implemented; model training pending**.
No model has learned these tasks yet and no strategic advantage is established.
They do not change the separately frozen budget experiment.

The first bounded pack enumerates all 256 two-player, two-action matrices with
binary integer payoffs. Each outcome has an ordered pair: row player's payoff,
then column player's payoff. Actions are 0 and 1. Five task types produce 2,816
short prompt/answer examples: payoff lookup, both players' best responses,
strictly dominant actions and all pure Nash equilibria. Ties return every best
response; strict dominance must beat the other action against both opponent
actions; a pure Nash equilibrium has no profitable unilateral deviation. `-`
means the requested solution set is empty. Matching-pennies examples have no
pure equilibrium; tied games can have multiple equilibria. Mixed strategies
are outside this introductory pack, so an empty pure set is not a claim that a
finite game has no mixed equilibrium.

`game_data.py` generates labels by finite unilateral-deviation enumeration.
`game_oracle.py` checks them using payoff maximization/regret and a strict prompt
parser. Sources are locally generated numeric facts/minimal symbols, with a
scoped admission containing author, procedure, source receipts and supporting
finite-domain evidence. The retained [manifest](../fixtures/game-data-v1/manifest.json)
binds the [dataset](../fixtures/game-data-v1/dataset.json). Auditing regenerates
all labels, carriers, identities, admission and family splits; contradictory
policy/evidence mutations reject independently of a model.

All row/column action relabellings and player-exchanged transposes are grouped
into one family before 80/10/remainder family-rank splitting. These are visible
conformance/development fixtures, not an unexposed confirmatory benchmark.
Training requires a separate frozen curriculum/learning protocol with exact
answer/EOS thresholds and exposure controls. Begin with payoffs/best responses,
then dominance/pure equilibria; introduce rational mixed-strategy/minimax games,
repeated play and uncertainty only after the simpler tasks learn reproducibly.
Binary payoffs alone do not cover general game theory or realistic incentives.

Offline standard-library audit:

```sh
PYTHONPATH=src python -c 'from pathlib import Path; from trinite.game_data import audit_bytes; p=Path("fixtures/game-data-v1"); print(audit_bytes((p/"dataset.json").read_bytes(), (p/"manifest.json").read_bytes())["counts"])'
PYTHONPATH=src python -m unittest discover -s tests -v
```

A later link to QEC could use a finite decoder-versus-error adversary game:
freeze allowed errors, decoder actions, correction/failure costs and budgets;
compute exact small-game baselines independently. That is a proposed robustness
experiment, not quantum advantage, proven physical protection or a substitute
for task correctness. See [research intake](RESEARCH.md#qec-geometry-etq-and-uft-id-intake).
