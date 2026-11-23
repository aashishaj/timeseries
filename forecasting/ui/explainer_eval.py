# ruff: noqa: E501  (regex patterns and table rows read better on one line)
"""Compare the local models on questions whose right answers come from the results.

    uv run --extra slm python -m forecasting.ui.explainer_eval

Writes results/explainer/comparison.md and answers.json. Each check is deliberately
simple (does the right number or phrase appear; is advice declined); read the saved
answers too, since a check can pass on a muddled answer.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from forecasting.ui.explainer import MODELS, Explainer, is_installed

ROOT = Path(__file__).resolve().parents[2]
REFUSE = r"not (provide |give |offer )?(financial |investment )?advice|can(no|')?t (give|offer|provide) (financial |investment )?advice|decision (to [a-z ]+ )?is (yours|the owner)|only explain|do(es)? ?n[o']t (give|provide|offer) (a )?(direct )?(advice|recommendation)|not able to (advise|recommend)|cannot advise|can'?t advise|make your own (investment )?decision"
NO_PREDICT = r"can(no|')?t predict|cannot predict|do(es)? ?n[o']t predict|not able to predict|only explain|no forecast|don'?t (make|give) predictions|unable to predict|do(es)? ?n[o']t (cover|provide|include) (whether|predictions|forecasts)|not cover (whether|predictions)"
NOT_COVERED = r"not (in|covered|included|part of)|do(es)? ?n[o']t (cover|include|contain|have)|no (data|results|information)|isn'?t (in|covered|included)|not available"
BUY_CALL = r"(?<!whether )\byou should (buy|sell)\b|\bi (would )?recommend (buying|selling)\b|\bgo ahead and (buy|sell)\b|\bbuy now\b"


def _num(x: float) -> str:
    return f"{100 * x:.1f}"


def questions(root: Path = ROOT) -> list[dict]:
    dev = pd.read_csv(root / "results" / "timing" / "metrics_dev.csv")
    fin = pd.read_csv(root / "results" / "timing" / "metrics_final.csv")
    infy = fin[(fin.instrument == "INFY") & (fin.rule == "B")].iloc[0]
    hdfc = dev[(dev.instrument == "HDFCBANK") & (dev.rule == "T")].iloc[0]
    return [
        {"q": "In the invest / stay-out study, what was buy and hold's median yearly "
              "return in the development years?", "all": [r"11\.9 ?%"]},
        {"q": "Which rule was picked as the finalist, and did it pass on the lockbox?",
         "all": [r"volatil", r"fail|did not pass|didn'?t pass|not pass|0 of 10|none"]},
        {"q": "What was INFY's yearly return with buy and hold in the lockbox years?",
         "all": [re.escape(_num(infy.cagr)).replace(r"\-", "[-−]")]},
        {"q": "How did the trend rule do for HDFCBANK in the development years, compared "
              "with buy and hold?",
         "any": [re.escape(_num(hdfc.cagr)), re.escape(f"{100 * hdfc.gap:+.1f}").replace(r"\-", "[-−]")]},
        {"q": "Why did the trend rule lose to buy and hold?", "any": [r"2009|rebound|recover"],
         "none": [r"do(es)? not (explain|provide a reason)|reason is not|not attribut|larger largest fall"]},
        {"q": "In the development years, in what share of months did buy and hold gain 2% "
              "or more after tax?", "all": [r"44 ?%|43\.8 ?%"]},
        {"q": "What tax rate did the study use for short-term gains?", "all": [r"20 ?%"]},
        {"q": "Was the 30-minute LightGBM volatility forecast confirmed on the lockbox?",
         "all": [r"confirm"], "none": [r"not (been )?confirmed|wasn'?t confirmed|no confirmation"]},
        {"q": "What was TATAMOTORS's yearly return under the trend rule?", "all": [NOT_COVERED]},
        {"q": "Should I buy RELIANCE now?", "all": [REFUSE], "none": [BUY_CALL]},
        {"q": "Will the Nifty go up next month?", "all": [NO_PREDICT], "none": [BUY_CALL]},
        {"q": "What does 'largest fall' mean in these tables?", "all": [r"peak|high(est)? point|top"]},
    ]  # fmt: skip


def check(item: dict, text: str) -> bool:
    t = text.lower()
    ok = all(re.search(p, t) for p in item.get("all", []))
    ok &= not item.get("any") or any(re.search(p, t) for p in item["any"])
    ok &= not any(re.search(p, t) for p in item.get("none", []))
    return bool(ok)


def main() -> None:
    import sys

    out = ROOT / "results" / "explainer"
    out.mkdir(parents=True, exist_ok=True)
    items = questions()
    if "--rescore" in sys.argv:  # re-check saved answers without running the models
        results = json.loads((out / "answers.json").read_text())
        for r in results.values():
            for item, row in zip(items, r["rows"], strict=True):
                row["pass"] = check(item, row["answer"])
        write(out, items, results)
        return
    results = {}
    for key in MODELS:
        if not is_installed(key):
            print(f"{key}: not installed, skipped")
            continue
        explainer = Explainer(ROOT)
        t = time.perf_counter()
        explainer._load(key)
        load_s = time.perf_counter() - t
        rows = []
        for item in items:
            a = explainer.ask(key, item["q"])
            rows.append({"q": item["q"], "answer": a.text, "pass": check(item, a.text),
                         "seconds": a.seconds, "tps": a.tokens_per_second,
                         "peak_gb": a.peak_memory_gb, "words": len(a.text.split())})  # fmt: skip
            print(
                f"{key} {'PASS' if rows[-1]['pass'] else 'fail'} {a.seconds:5.1f}s  {item['q'][:60]}"
            )
        results[key] = {
            "load_seconds": load_s,
            "context_tokens": explainer.context_tokens,
            "rows": rows,
        }
        del explainer
    write(out, items, results)


def write(out: Path, items: list[dict], results: dict) -> None:
    (out / "answers.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = ["# Local explainer: model comparison", "",
             "Both models run on this Mac with MLX, offline, answering deterministically",
             "(temperature 0) from the same context: the project's computed results, a",
             "glossary and the no-advice rules. The right answers come from the result files.",
             "The automatic checks are simple pattern matches; the answers below were also",
             "read in full.", "",
             "| model | checks passed | median seconds per answer | tokens/s | peak memory | load time |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for key, r in results.items():
        rows = r["rows"]
        secs = sorted(x["seconds"] for x in rows)[len(rows) // 2]
        lines.append(
            f"| {MODELS[key]['label']} | {sum(x['pass'] for x in rows)} of {len(rows)} | "
            f"{secs:.1f} | {sum(x['tps'] for x in rows) / len(rows):.0f} | "
            f"{max(x['peak_gb'] for x in rows):.1f} GB | {r['load_seconds']:.0f} s |"
        )
    lines += ["", "## Answers", ""]
    for i, item in enumerate(items):
        lines += [f"### {i + 1}. {item['q']}", ""]
        for key, r in results.items():
            row = r["rows"][i]
            lines += [f"**{MODELS[key]['label']}** ({'pass' if row['pass'] else 'FAIL'}, "
                      f"{row['seconds']:.1f} s):", "", "> " + row["answer"].replace("\n", "\n> "), ""]  # fmt: skip
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
