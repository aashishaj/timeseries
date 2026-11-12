# Leaderboard

Latest result per model configuration. `trials` counts every distinct configuration
ever run on that data/split/horizon (losers included). The more trials, the more
a top result may be luck, so the lockbox (`split = final`) is the number that counts.

## synthetic / dev / 30min (15 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| combo_opt | 0.412 | 0.519 | 0.742 | 0 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt` | 2026-11-11 11:07 |
| lgbm | 0.412 | 0.522 | 0.731 | 0 | 10/10 | `lightgbm` | 2026-11-12 07:20 |
| combo_equal | 0.436 | 0.554 | 0.746 | 0 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal` | 2026-11-11 11:07 |
| gjr_garch | 0.463 | 0.577 | 0.782 | 0 | 10/10 | `garch, asymmetric=True` | 2026-11-11 10:44 |
| garch | 0.463 | 0.580 | 0.784 | 0 | 10/10 | `garch, asymmetric=False` | 2026-11-11 11:07 |
| arma_logrv_mix | 0.470 | 0.591 | 0.775 | 0 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-12 07:20 |
| har_mix | 0.474 | 0.599 | 0.770 | 0 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-11 11:07 |
| ewma_hl1 | 0.486 | 0.609 | 0.805 | 8.6e-320 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-11 10:44 |
| nbeats | 0.506 | 0.627 | 0.799 | 5.5e-316 | 10/10 | `nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64` | 2026-11-12 07:20 |
| ewma_hl5 | 0.534 | 0.667 | 0.829 | 8.4e-244 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-11 10:44 |
| har_pooled | 0.548 | 0.678 | 0.828 | 4.3e-213 | 10/10 | `har, pooled=True` | 2026-11-11 10:44 |
| har | 0.556 | 0.679 | 0.834 | 1.1e-200 | 10/10 | `har, pooled=False` | 2026-11-11 10:44 |
| hist_mean | 0.637 | 0.780 | 0.914 | 2.2e-116 | 10/10 | `historical_mean` | 2026-11-11 10:44 |
| seasonal_mean_5d | 0.661 | 0.763 | 0.912 | 2.5e-110 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-11 11:07 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-12 07:20 |

## synthetic / dev / next_session (15 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| combo_equal | 0.638 | 0.650 | 0.793 | 4.7e-21 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal` | 2026-11-11 11:07 |
| combo_opt | 0.638 | 0.645 | 0.823 | 2.3e-18 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt` | 2026-11-11 11:07 |
| arma_logrv_mix | 0.644 | 0.655 | 0.814 | 6.7e-21 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-12 07:20 |
| lgbm | 0.664 | 0.657 | 0.817 | 2.8e-14 | 10/10 | `lightgbm` | 2026-11-12 07:20 |
| har_mix | 0.674 | 0.681 | 0.814 | 1.4e-17 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-11 11:07 |
| har_pooled | 0.678 | 0.687 | 0.817 | 7.3e-17 | 10/10 | `har, pooled=True` | 2026-11-11 10:44 |
| seasonal_mean_5d | 0.696 | 0.686 | 0.843 | 7.7e-13 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-11 11:07 |
| har | 0.708 | 0.692 | 0.831 | 2.6e-14 | 10/10 | `har, pooled=False` | 2026-11-11 10:44 |
| ewma_hl1 | 0.712 | 0.714 | 0.847 | 7.2e-25 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-11 10:44 |
| ewma_hl5 | 0.718 | 0.723 | 0.850 | 5.5e-08 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-11 10:44 |
| garch | 0.813 | 0.788 | 0.909 | 9.3e-06 | 10/10 | `garch, asymmetric=False` | 2026-11-11 11:07 |
| gjr_garch | 0.814 | 0.788 | 0.911 | 2.7e-05 | 10/10 | `garch, asymmetric=True` | 2026-11-11 10:44 |
| hist_mean | 0.970 | 0.955 | 0.978 | 0.84 | 8/10 | `historical_mean` | 2026-11-11 10:44 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-12 07:20 |
| nbeats | 1.036 | 0.968 | 1.044 | 0.4 | 4/10 | `nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64` | 2026-11-12 07:20 |
