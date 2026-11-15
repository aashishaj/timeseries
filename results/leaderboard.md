# Leaderboard

Latest result per model configuration. `trials` counts every distinct configuration
ever run on that data/split/horizon (losers included). The more trials, the more
a top result may be luck, so the lockbox (`split = final`) is the number that counts.

## real:kite / dev / 30min (17 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| combo_opt | 0.042 | 0.291 | 0.602 | 0.26 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt` | 2026-11-12 17:12 |
| lgbm | 0.042 | 0.291 | 0.588 | 0.26 | 10/10 | `lightgbm` | 2026-11-12 17:12 |
| combo_equal | 0.043 | 0.289 | 0.600 | 0.26 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal` | 2026-11-12 17:12 |
| ewma_hl1 | 0.044 | 0.300 | 0.611 | 0.26 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-12 17:12 |
| garch | 0.044 | 0.325 | 0.632 | 0.26 | 10/10 | `garch, asymmetric=False` | 2026-11-12 17:12 |
| gjr_garch | 0.044 | 0.329 | 0.633 | 0.26 | 10/10 | `garch, asymmetric=True` | 2026-11-12 17:12 |
| arma_logrv_mix | 0.046 | 0.298 | 0.632 | 0.26 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-12 17:12 |
| ewma_hl5 | 0.046 | 0.300 | 0.624 | 0.26 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-12 17:12 |
| har_mix | 0.046 | 0.299 | 0.654 | 0.26 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-12 17:12 |
| nbeats_deep | 0.048 | 0.301 | 0.628 | 0.26 | 10/10 | `nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128` | 2026-11-12 17:12 |
| har | 0.049 | 0.302 | 0.699 | 0.26 | 10/10 | `har, pooled=False` | 2026-11-12 17:12 |
| har_pooled | 0.049 | 0.303 | 0.700 | 0.26 | 10/10 | `har, pooled=True` | 2026-11-12 17:12 |
| nbeats | 0.050 | 0.317 | 0.660 | 0.27 | 10/10 | `nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64` | 2026-11-12 17:12 |
| seasonal_mean_5d | 0.059 | 0.346 | 0.687 | 0.27 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-12 17:12 |
| hist_mean | 0.065 | 0.346 | 0.847 | 0.27 | 10/10 | `historical_mean` | 2026-11-12 17:12 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-12 17:12 |
| chronos_tiny | nan | nan | nan | - | nan | `chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100` | 2026-11-12 17:12 |

## real:kite / dev / next_session (17 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| combo_equal | 0.607 | 0.575 | 0.786 | 3e-13 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal` | 2026-11-12 17:12 |
| arma_logrv_mix | 0.611 | 0.576 | 0.791 | 6.8e-12 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-12 17:12 |
| combo_opt | 0.612 | 0.636 | 0.843 | 2.5e-11 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt` | 2026-11-12 17:12 |
| har | 0.637 | 0.604 | 0.862 | 7.5e-09 | 10/10 | `har, pooled=False` | 2026-11-12 17:12 |
| har_pooled | 0.640 | 0.606 | 0.862 | 2.9e-09 | 10/10 | `har, pooled=True` | 2026-11-12 17:12 |
| har_mix | 0.640 | 0.606 | 0.862 | 2.9e-09 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-12 17:12 |
| ewma_hl5 | 0.648 | 0.595 | 0.785 | 1.2e-08 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-12 17:12 |
| seasonal_mean_5d | 0.668 | 0.638 | 0.850 | 9.3e-08 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-12 17:12 |
| lgbm | 0.674 | 0.645 | 0.863 | 7.9e-10 | 10/10 | `lightgbm` | 2026-11-12 17:12 |
| ewma_hl1 | 0.679 | 0.628 | 0.786 | 6.7e-12 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-12 17:12 |
| nbeats_deep | 0.767 | 0.636 | 0.866 | 0.00028 | 10/10 | `nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128` | 2026-11-12 17:12 |
| gjr_garch | 0.777 | 1.059 | 0.892 | 0.00094 | 9/10 | `garch, asymmetric=True` | 2026-11-12 17:12 |
| garch | 0.780 | 0.992 | 0.890 | 0.0013 | 9/10 | `garch, asymmetric=False` | 2026-11-12 17:12 |
| hist_mean | 0.797 | 0.647 | 0.851 | 0.013 | 8/10 | `historical_mean` | 2026-11-12 17:12 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-12 17:12 |
| nbeats | 1.024 | 0.930 | 1.032 | 0.74 | 4/10 | `nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64` | 2026-11-12 17:12 |
| chronos_tiny | nan | nan | nan | - | nan | `chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100` | 2026-11-12 17:12 |

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
