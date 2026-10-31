# Fragile-proof detection: summary

- solver: `z3[Z3 version 4.12.2 - 64 bit]`; timeout 10.0 s; success = `unsat`
- queries analysed: 268 in 5 projects (manifest 268, incomplete 0, dropped for original-query error 0, variant errors skipped 0)
- labels from independent reference variants (fragile = success rate in [0.1, 0.9]): {'pure': 177.0, 'borderline': 25.0, 'fragile': 66.0}
- detectable from the pool at all (the ceiling): {'pure': 16.0, 'borderline': 12.0, 'fragile': 61.0}
- exhaustive pool testing would cost 4.72 CPU hours (the x-axis scale)
- seeds 20, project-held-out folds 3, tuned: True

## Main result

| strategy | AUDC (mean, 95% over seeds) | budget to 50% of ceiling | budget to 80% | fragile found @10% / @25% budget | false alarms @25% |
|---|---|---|---|---|---|
| uniform | 0.810 [0.785, 0.840] | 0.16 | 0.27 | 0.22 / 0.76 | 5.0 |
| random | 0.803 [0.768, 0.838] | 0.18 | 0.29 | 0.25 / 0.72 | 5.4 |
| runtime | 0.779 [0.756, 0.812] | 0.18 | 0.37 | 0.33 / 0.64 | 8.8 |
| adaptive | 0.835 [0.813, 0.863] | 0.14 | 0.25 | 0.31 / 0.80 | 6.8 |
| adaptive-tuned | 0.833 [0.813, 0.861] | 0.14 | 0.25 | 0.32 / 0.80 | 7.5 |
| adaptive-nopool | 0.832 [0.808, 0.861] | 0.15 | 0.25 | 0.31 / 0.80 | 6.8 |
| adaptive-noboost | 0.832 [0.810, 0.861] | 0.14 | 0.25 | 0.31 / 0.79 | 6.7 |
| adaptive-noexplore | 0.839 [0.819, 0.866] | 0.14 | 0.25 | 0.33 / 0.80 | 7.2 |

Budgets are fractions of the cost of testing every pool variant of every query (1.0 = exhaustive).

## Paired comparison with uniform testing

| strategy | AUDC difference | 95% over seeds | seeds better | 95% over projects |
|---|---|---|---|---|
| random | -0.0070 | [-0.0248, 0.0063] | 5/20 | only 5 projects: project-level CI not meaningful (need >= 8) |
| runtime | -0.0315 | [-0.0543, -0.0113] | 0/20 | only 5 projects: project-level CI not meaningful (need >= 8) |
| adaptive | 0.0250 | [0.0143, 0.0331] | 20/20 | only 5 projects: project-level CI not meaningful (need >= 8) |
| adaptive-tuned | 0.0231 | [0.0129, 0.0305] | 20/20 | only 5 projects: project-level CI not meaningful (need >= 8) |
| adaptive-nopool | 0.0220 | [0.0115, 0.0302] | 20/20 | only 5 projects: project-level CI not meaningful (need >= 8) |
| adaptive-noboost | 0.0218 | [0.0101, 0.0310] | 20/20 | only 5 projects: project-level CI not meaningful (need >= 8) |
| adaptive-noexplore | 0.0282 | [0.0166, 0.0370] | 20/20 | only 5 projects: project-level CI not meaningful (need >= 8) |

Seeds capture variation of test order and strategy randomness; the project interval captures which projects were sampled (approximate: allocation was not re-run on each resample).

## Agreement with the benchmark's curated classification

Rows: the benchmark's own group (curated by its authors with a different solver build and timeout).
Columns: how THIS experiment's independent reference labels those queries.

| curated group | pure | borderline | fragile | fragile share |
|---|---|---|---|---|
| stable_core | 80 | 1 | 0 | 0% |
| stable_ext | 50 | 15 | 20 | 24% |
| unstable_core | 6 | 0 | 8 | 57% |
| unstable_ext | 41 | 9 | 38 | 43% |

If curated-unstable groups show a clearly higher fragile share than curated-stable groups, the labelling and the variant generator are behaving sensibly on real data. If not, investigate before trusting any strategy comparison.

## Parameters chosen on training projects

- fold 1: {'eps': 0.05, 'gamma': 3.0, 'kappa': 25.0}
- fold 2: {'eps': 0.05, 'gamma': 0.0, 'kappa': 5.0}
- fold 3: {'eps': 0.05, 'gamma': 3.0, 'kappa': 5.0}

## Measurement controls

- 268 unchanged originals were re-run; 4 changed outcome with identical input (these are timing/limit noise, not input sensitivity); median relative runtime spread 0.051.
- queries with both `sat` and `unsat` among equivalent variants (soundness-bug candidates OR a transformation bug - investigate before claiming anything): 0 []

## Figures

- fig1_detection_vs_cpu.png
- fig2_audc_difference.png

## How to read this honestly

- A strategy only 'wins' if its curve is above uniform on held-out projects with intervals that exclude zero.
- Witnesses found here are measurements; confirm them with `python -m fragile confirm` (re-runs) before calling a query fragile.
- Nothing here shows that unflagged queries are stable; the ceiling is limited by the pool size.
