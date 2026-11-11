# Leaderboard

Latest result per model configuration. `trials` counts every distinct configuration
ever run on that data/split/horizon (losers included). The more trials, the more
a top result may be luck, so the lockbox (`split = final`) is the number that counts.

## synthetic / dev / 30min (11 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| garch | 0.463 | 0.576 | 0.782 | 0 | 10/10 | `garch, asymmetric=False` | 2026-11-11 10:30 |
| gjr_garch | 0.463 | 0.577 | 0.782 | 0 | 10/10 | `garch, asymmetric=True` | 2026-11-11 10:30 |
| arma_logrv_mix | 0.469 | 0.588 | 0.774 | 0 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-11 10:30 |
| har_mix | 0.474 | 0.597 | 0.769 | 0 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-11 10:30 |
| ewma_hl1 | 0.486 | 0.609 | 0.805 | 8.6e-320 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-11 10:30 |
| ewma_hl5 | 0.534 | 0.667 | 0.829 | 8.4e-244 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-11 10:30 |
| har_pooled | 0.548 | 0.678 | 0.828 | 4.3e-213 | 10/10 | `har, pooled=True` | 2026-11-11 10:30 |
| har | 0.556 | 0.679 | 0.834 | 1.1e-200 | 10/10 | `har, pooled=False` | 2026-11-11 10:30 |
| hist_mean | 0.637 | 0.780 | 0.914 | 2.2e-116 | 10/10 | `historical_mean` | 2026-11-11 10:30 |
| seasonal_mean_5d | 0.661 | 0.759 | 0.909 | 5.9e-107 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-11 10:30 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-11 10:30 |

## synthetic / dev / next_session (11 trials)

| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |
|---|---|---|---|---|---|---|---|
| arma_logrv_mix | 0.645 | 0.657 | 0.814 | 7.4e-20 | 10/10 | `arma_logrv, intraday_beta=0.5, order=[1, 1]` | 2026-11-11 10:30 |
| har_pooled | 0.678 | 0.687 | 0.817 | 7.3e-17 | 10/10 | `har, pooled=True` | 2026-11-11 10:30 |
| har_mix | 0.678 | 0.687 | 0.817 | 7.3e-17 | 10/10 | `har, intraday_beta=0.5, pooled=True` | 2026-11-11 10:30 |
| seasonal_mean_5d | 0.700 | 0.693 | 0.846 | 3.4e-12 | 10/10 | `seasonal_naive, n_sessions=5` | 2026-11-11 10:30 |
| har | 0.708 | 0.692 | 0.831 | 2.6e-14 | 10/10 | `har, pooled=False` | 2026-11-11 10:30 |
| ewma_hl1 | 0.712 | 0.714 | 0.847 | 7.2e-25 | 10/10 | `ewma, halflife_sessions=1.0` | 2026-11-11 10:30 |
| ewma_hl5 | 0.718 | 0.723 | 0.850 | 5.5e-08 | 10/10 | `ewma, halflife_sessions=5.0` | 2026-11-11 10:30 |
| garch | 0.814 | 0.788 | 0.911 | 2.2e-05 | 10/10 | `garch, asymmetric=False` | 2026-11-11 10:30 |
| gjr_garch | 0.814 | 0.788 | 0.911 | 2.7e-05 | 10/10 | `garch, asymmetric=True` | 2026-11-11 10:30 |
| hist_mean | 0.970 | 0.955 | 0.978 | 0.84 | 8/10 | `historical_mean` | 2026-11-11 10:30 |
| naive | 1.000 | 1.000 | 1.000 | - | - | `random_walk` | 2026-11-11 10:30 |
