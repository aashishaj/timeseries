# Leaderboard

Latest result per model configuration. `trials` counts every distinct configuration
ever run on that data/split/horizon (losers included). The more trials, the more
a top result may be luck, so the lockbox (`split = final`) is the number that counts.

## real:kite / dev / 30min (17 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| combo_opt | 0.329 | 0.262 | 0.584 | 9.8e-55 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt` | 2026-11-15 09:15 |
| lgbm | 0.329 | 0.263 | 0.571 | 8.5e-55 | 10/10 | `lightgbm` | 2026-11-15 09:15 |
| combo_equal | 0.338 | 0.261 | 0.585 | 6.3e-53 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal` | 2026-11-15 09:15 |
| garch | 0.346 | 0.296 | 0.615 | 3e-52 | 10/10 | `garch, asymmetric=False` | 2026-11-15 09:15 |
| gjr_garch | 0.347 | 0.300 | 0.616 | 3.2e-52 | 10/10 | `garch, asymmetric=True` | 2026-11-15 09:15 |
| ewma_hl1 | 0.347 | 0.272 | 0.595 | 6.4e-51 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-15 09:15 |
| arma_logrv_mix | 0.367 | 0.269 | 0.618 | 4.1e-48 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-15 09:15 |
| ewma_hl5 | 0.367 | 0.271 | 0.608 | 4.8e-48 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-15 09:15 |
| har_mix | 0.369 | 0.271 | 0.639 | 9.5e-48 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-15 09:15 |
| chronos_tiny | 0.378 | 0.273 | 0.588 | 6.4e-47 | 10/10 | `chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100` | 2026-11-15 09:15 |
| nbeats_deep | 0.379 | 0.273 | 0.612 | 7.5e-47 | 10/10 | `nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128` | 2026-11-15 09:15 |
| har | 0.391 | 0.274 | 0.685 | 8.6e-44 | 10/10 | `har, pooled=False` | 2026-11-15 09:15 |
| har_pooled | 0.392 | 0.274 | 0.686 | 1.1e-43 | 10/10 | `har, pooled=True` | 2026-11-15 09:15 |
| nbeats | 0.397 | 0.289 | 0.646 | 3.3e-44 | 10/10 | `nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64` | 2026-11-15 09:15 |
| seasonal_mean_5d | 0.467 | 0.312 | 0.666 | 1.7e-33 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-15 09:15 |
| hist_mean | 0.519 | 0.318 | 0.843 | 7.5e-28 | 10/10 | `historical_mean` | 2026-11-15 09:15 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-15 09:15 |

## real:kite / dev / next_session (17 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| combo_equal | 0.619 | 0.573 | 0.792 | 4e-16 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal` | 2026-11-15 09:15 |
| arma_logrv_mix | 0.627 | 0.578 | 0.798 | 1.8e-15 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-15 09:15 |
| combo_opt | 0.648 | 0.616 | 0.837 | 5.6e-15 | 10/10 | `combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt` | 2026-11-15 09:15 |
| har | 0.657 | 0.605 | 0.876 | 7.7e-11 | 10/10 | `har, pooled=False` | 2026-11-15 09:15 |
| har_pooled | 0.661 | 0.607 | 0.875 | 2.1e-11 | 10/10 | `har, pooled=True` | 2026-11-15 09:15 |
| har_mix | 0.661 | 0.607 | 0.875 | 2.1e-11 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-15 09:15 |
| lgbm | 0.675 | 0.628 | 0.852 | 4.4e-12 | 10/10 | `lightgbm` | 2026-11-15 09:15 |
| ewma_hl5 | 0.697 | 0.603 | 0.794 | 2e-09 | 9/10 | `ewma, halflife_sessions=5.0` | 2026-11-15 09:15 |
| seasonal_mean_5d | 0.724 | 0.648 | 0.866 | 6.8e-08 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-15 09:15 |
| ewma_hl1 | 0.730 | 0.656 | 0.796 | 8e-13 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-15 09:15 |
| gjr_garch | 0.781 | 1.057 | 0.901 | 0.00044 | 10/10 | `garch, asymmetric=True` | 2026-11-15 09:15 |
| garch | 0.784 | 0.994 | 0.900 | 0.00043 | 10/10 | `garch, asymmetric=False` | 2026-11-15 09:15 |
| chronos_tiny | 0.790 | 0.629 | 0.777 | 0.0002 | 9/10 | `chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100` | 2026-11-15 09:15 |
| nbeats_deep | 0.828 | 0.635 | 0.855 | 0.0029 | 8/10 | `nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128` | 2026-11-15 09:15 |
| hist_mean | 0.830 | 0.650 | 0.904 | 0.023 | 8/10 | `historical_mean` | 2026-11-15 09:15 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-15 09:15 |
| nbeats | 1.025 | 0.928 | 1.002 | 0.7 | 5/10 | `nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64` | 2026-11-15 09:15 |

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
