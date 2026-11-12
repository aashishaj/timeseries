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

Kite access tokens expire every morning. Run this yourself in Terminal on the Mac;
never paste the API secret into chat or a file in the repo:

```bash
uv run forecasting kite-login
```

1. It asks for your API key, then your API secret (typing is hidden, and nothing
   lands in shell history).
2. Your browser opens the Zerodha login. Log in as usual.
3. Zerodha redirects to your app's redirect URL (`http://127.0.0.1:8766/callback`);
   `kite-login` is listening there, catches the login, and says it is done. If that
   port is busy (say your live pipeline uses it), it asks you to paste the URL from
   the browser's address bar instead.

The access token and API key are saved to `~/.config/nse-forecasting/` (readable
only by you) and never printed. The secret is used once and not stored, unless you
run `uv run forecasting kite-login --remember-secret`: then it is kept in the same
private folder and you are not asked again. Nothing from that folder is ever
committed: it lives outside the repo.

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

## Keep the Mac working with the lid closed

A MacBook normally sleeps when the lid closes, and `caffeinate` does not stop that.
To keep jobs (and a Remote Control session) running with the lid shut:

```bash
sudo pmset -a disablesleep 1      # no sleep at all, lid closed included
pmset -g | grep -i sleep          # check: SleepDisabled 1
```

Turn it back off when you are done, or before putting the Mac in a bag:

```bash
sudo pmset -a disablesleep 0
```

While it is on: keep the Mac plugged in, on a hard surface (it cannot cool well
under a pillow or in a bag), and leave the Claude app or Terminal session running.
Wi-Fi stays connected. A GUI alternative is the free Amphetamine app, which has a
closed-lid option. Clamshell mode (external display + power) also works without
changing settings.

## Drive it from your phone

Run Claude on the Mac in the repo folder: either the Claude desktop app with the
folder shared under Settings > Claude Code > Remote Control, or `claude
remote-control` in a Terminal inside the folder. The session then appears in the
Claude app on your phone. Good first message for that session:

> Read README.md, PROGRESS.md and REAL_DATA.md. Set up the project per REAL_DATA.md
> (uv, `uv sync --extra kite`). Do not ask me for Kite secrets: I run
> `forecasting kite-login` myself. Then run backfill-kite and run-all, and commit
> and push the results folder.
