# Evaluation report (synthetic data)

Trained on synthetic (simulated) data. Metrics show how well the models recover the rules written in dummydataset.py, not how well they would predict a real person.

Hold-out: 4 people never seen in training (1940 rows). The hold-out contains only 4 people, so hold-out metrics are noisy (they change a lot with a different split). Prefer the cross-validated numbers, or generate more people in dummydataset.py.

## Regression (xgboost)
- Hold-out R2 = 0.005, MAE = 9.19 points (score range 0-100)
- Cross-validated numbers (more stable) are in metadata.json under regression.candidates

## Classification (xgboost)
- Hold-out accuracy = 0.605, macro one-vs-rest ROC-AUC = 0.762
- Read confusion_matrix.png: the 'low' level is rare, so expect it to be the hardest class.

## Permutation importance (hold-out)
| feature | regression MAE increase | classification F1 drop |
|---|---|---|
| stress_level | 1.138 | 0.044 |
| study_hours | 0.456 | 0.024 |
| caffeine_mg | 0.307 | 0.001 |
| study_sessions | 0.133 | 0.005 |
| journaled | -0.123 | -0.001 |

Importance describes what the fitted model uses. It is not causal and correlated features share credit.
