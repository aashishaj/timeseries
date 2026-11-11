# Running on real NSE data (macOS)

Everything runs on your Mac. Your Kite API key, secret and access token stay on
the Mac: never in this repo, never in chat, never in the cloud.

## Once: tools and code (about 5 minutes)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # Python tool manager
git clone https://github.com/aashishaj/timeseries && cd timeseries
git checkout claude/nse-volatility-forecasting-w1h02e
uv sync --extra kite                                    # installs everything
```

If LightGBM later complains about `libomp`: `brew install libomp`.

## Each day you use Kite: log in (you, in Terminal)

Kite access tokens expire every morning. Type these yourself; do not paste the
secret anywhere else:

```bash
export KITE_API_KEY=your_api_key
export KITE_API_SECRET=your_api_secret
uv run forecasting kite-login
```

It prints a Zerodha login URL. Log in, copy the URL you are redirected to, and
paste it back. The token is saved to `~/.config/nse-forecasting/` (readable only
by you) and is never printed.

## Get data and run everything

```bash
uv run forecasting backfill-kite --out data/kite          # 1 year of 1-min candles
uv run forecasting run-all --data-root data/kite          # ~20-40 min
```

* `backfill-kite` takes the symbols from `forecasting/configs/symbols.yaml`, asks
  Kite for 55-day chunks (Kite's limit for minute candles is 60), and resumes if
  interrupted. Use `--start 2024-11-01` for more history.
* `run-all` checks the data, backtests every model on the development period,
  simulates the paper trading, and writes `results/run_kite/REPORT.md`.
* The lockbox (last 40 sessions) stays sealed. Open it once, at the very end,
  with `uv run forecasting backtest --config forecasting/configs/rv_ladder.yaml
  --data-root data/kite --final`.

## Share the results

`data/` is git-ignored, so candles are never committed. The report is small:

```bash
git add results/run_kite results/leaderboard.*
git commit -m "Real-data results" && git push
```

## Keep the Mac awake while it runs

Plugged in, lid open, and either System Settings > Battery > Options > "Prevent
automatic sleeping on power adapter when the display is off", or run
`caffeinate -i` in a spare Terminal window.
