"""A small language model, run locally with MLX, that explains this project's results.

The model sees only material the project computed (the study tables, the progress log's
findings, the README's results section) and is told to answer from that alone, to say
when something is not in it, and never to predict prices or advise buying or selling.
Nothing leaves the Mac: the model is loaded from the local Hugging Face cache.

The fixed context is processed once per model into a cache; each question then only
needs its own tokens processed, and the cache is trimmed back afterwards.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MODELS = {
    "qwen3-4b": {
        "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "label": "Qwen3 4B",
        "size_gb": 2.28,
    },
    "qwen3-8b": {
        "repo": "mlx-community/Qwen3-8B-4bit",
        "label": "Qwen3 8B",
        "size_gb": 4.62,
    },
}
MAX_QUESTION_CHARS = 1000
FILES = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]  # what a model needs

SYSTEM_PROMPT = """You explain the results of a personal research project about NSE \
(Indian) stocks to its owner, who is not a statistician. The project's own results are \
below, between <results> and </results>.

Rules:
1. Answer only from the results below. Quote numbers exactly as they appear there, \
with their units: never add or drop decimal places, never combine or average numbers \
yourself, and never attach one stock's number to another stock or rule.
2. If the answer is not in the results, say plainly that the results do not cover it. \
Do not guess.
3. Never predict prices, returns or market moves, and never advise buying, selling or \
holding anything. If asked, say that you only explain past results, that they are not \
advice, and that the decision is the owner's; then explain what the results show.
4. Use plain English, short sentences, and explain any technical term in a few words. \
Keep answers under 200 words unless asked for more.

<results>
{context}
</results>"""


def local_path(key: str) -> Path | None:
    """The model's folder in the local Hugging Face cache, or None if not downloaded.
    Never contacts the network."""
    try:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(MODELS[key]["repo"], local_files_only=True, allow_patterns=FILES)
        )
    except Exception:
        return None
    return path if any(path.glob("*.safetensors")) else None


def is_installed(key: str) -> bool:
    return local_path(key) is not None


def runtime_available() -> bool:
    try:
        import mlx_lm  # noqa: F401

        return True
    except Exception:
        return False


def download(key: str) -> str:
    from huggingface_hub import snapshot_download

    repo = MODELS[key]["repo"]
    log.info("downloading %s (about %.1f GB) from huggingface.co", repo, MODELS[key]["size_gb"])
    snapshot_download(repo, allow_patterns=FILES)
    return f"Done: {MODELS[key]['label']} is installed."


# ---------------------------------------------------------------------- context
def _section(text: str, start: str, stop: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(stop, i + len(start))
    return text[i : j if j >= 0 else len(text)].strip()


def _pct(x) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.1f}%"


GLOSSARY = """# What the terms mean
- Yearly return: growth per year, compounded, after charges and tax, as if everything \
were sold on the last day.
- vs B / vs buy and hold: a rule's yearly return minus buy and hold's, in percentage \
points. Negative means the rule did worse.
- Largest fall: the biggest drop in an account's value from its previous highest point \
(its peak) to a later low. A smaller fall means less pain along the way.
- Worst year: the account's worst calendar-year return.
- Months +2%: the share of months in which the account gained 2% or more after tax.
- Trades/yr: buy-then-sell round trips per year.
- Development years: the earlier years, used to compare the rules and pick a finalist.
- Lockbox: the most recent years, kept unseen until the rules were final, then checked \
once to confirm or reject the finalist.
- Buy and hold (B): buy on the first day and never sell until the end.
- QLIKE: a score for volatility forecasts; lower is better; "0.299 x naive" means much \
better than simply repeating the last value.

# Key facts
- In the development years the timing rules usually had smaller largest falls than buy \
and hold, but much lower yearly returns; the drop in return outweighed the smaller falls.
- On the lockbox the finalist (the volatility rule) had lower returns and deeper falls \
than buy and hold.
"""


def build_context(root: Path) -> str:
    """The project's computed results, as compact text for the model."""
    from forecasting.timing.study import LABELS

    parts = [GLOSSARY]
    readme = (root / "README.md").read_text() if (root / "README.md").exists() else ""
    progress = (root / "PROGRESS.md").read_text() if (root / "PROGRESS.md").exists() else ""
    if section := _section(readme, "## Results on real data", "\n## Layout"):
        parts.append("# Project results (from the README)\n" + section)
    if section := _section(progress, "### 2026-11-23, round 9", "\n### "):
        parts.append(
            "# Invest / stay-out study: plan and findings (from the progress log)\n" + section
        )
    for split, title in (("dev", "development years"), ("final", "lockbox (last 3 years)")):
        path = root / "results" / "timing" / f"metrics_{split}.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path)
        summary = [
            f"# Invest / stay-out study, summary per rule, {title}: {t['start'].min()} to "
            f"{t['end'].max()}. Medians and averages are over the {t['instrument'].nunique()} "
            "accounts (10 stocks and NIFTYBEES). Quote these for 'overall' or 'on average' "
            "questions.",
            "rule | median yearly return | median vs B | median largest fall | average "
            "months +2% | average trades/yr | accounts passing",
        ]
        for rule, g in t.groupby("rule", sort=False):
            gap = "" if rule == "B" else f"{100 * g['gap'].median():+.1f} pts"
            passing = "" if rule == "B" else f"{int(g['passes'].sum())} of {len(g)}"
            summary.append(
                f"{LABELS.get(rule, rule)} | {_pct(g['cagr'].median())} | {gap} | "
                f"{_pct(g['max_drawdown'].median())} | {_pct(g['months_2pct'].mean())} | "
                f"{g['trades_per_year'].mean():.1f} | {passing}"
            )
        parts.append("\n".join(summary))
        lines = [
            f"# Invest / stay-out study, every account, {title}: {t['start'].min()} to "
            f"{t['end'].max()}. After charges and tax. 'vs B' = yearly return minus buy "
            "and hold's, in percentage points.",
            "instrument | rule | yearly return | vs B | largest fall | worst year | "
            "months +2% | trades/yr | passes",
        ]
        for r in t.itertuples():
            gap = "" if r.rule == "B" else f"{100 * r.gap:+.1f} pts"
            passes = "" if r.rule == "B" else ("yes" if r.passes else "no")
            lines.append(
                f"{r.instrument} | {LABELS.get(r.rule, r.rule)} | {_pct(r.cagr)} | {gap} | "
                f"{_pct(r.max_drawdown)} | {_pct(r.worst_year)} | {_pct(r.months_2pct)} | "
                f"{r.trades_per_year:.1f} | {passes}"
            )
        parts.append("\n".join(lines))
    text = "\n\n".join(parts)
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------------------- answering
@dataclass
class Answer:
    question: str
    model: str
    text: str = ""
    done: bool = False
    error: str | None = None
    started: float = field(default_factory=time.time)
    seconds: float = 0.0
    tokens_per_second: float = 0.0
    peak_memory_gb: float = 0.0

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "model": MODELS[self.model]["label"],
            "text": self.text,
            "done": self.done,
            "error": self.error,
            "seconds": round(self.seconds or time.time() - self.started, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "peak_memory_gb": round(self.peak_memory_gb, 1),
        }


class Explainer:
    """Loads one model at a time and answers one question at a time."""

    def __init__(self, root: Path, max_tokens: int = 600, temperature: float = 0.0) -> None:
        self.root = Path(root)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.key: str | None = None
        self.model = self.tokenizer = self.cache = None
        self.prefix: list[int] = []
        self.context_tokens = 0
        self._lock = threading.Lock()

    def _messages(self, question: str) -> list[dict]:
        system = SYSTEM_PROMPT.format(context=build_context(self.root))
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def _render(self, messages: list[dict], generation: bool) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=generation, enable_thinking=False, tokenize=True
        )

    def _load(self, key: str) -> None:
        if self.key == key:
            return
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.model = self.tokenizer = self.cache = None
        gc.collect()
        mx.clear_cache()
        path = local_path(key)
        if path is None:
            raise RuntimeError(f"{MODELS[key]['label']} is not downloaded")
        log.info("loading %s", MODELS[key]["label"])
        self.model, self.tokenizer = load(str(path))  # local folder: no network calls
        self.key = key
        # process the fixed part (system prompt + results) once
        system_only = self._messages("")[:1]
        self.prefix = list(self._render(system_only, generation=False))
        self.cache = make_prompt_cache(self.model)
        tokens = mx.array(self.prefix)
        for i in range(0, len(self.prefix), 512):
            self.model(tokens[None, i : i + 512], cache=self.cache)
            mx.eval([c.state for c in self.cache])
        self.context_tokens = len(self.prefix)
        log.info("%s ready; results context is %d tokens", MODELS[key]["label"], len(self.prefix))

    def ask(self, key: str, question: str, on_text: Callable[[str], None] | None = None) -> Answer:
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        question = question.strip()[:MAX_QUESTION_CHARS]
        answer = Answer(question=question, model=key)
        with self._lock:
            self._load(key)
            full = list(self._render(self._messages(question), generation=True))
            use_cache = full[: len(self.prefix)] == self.prefix
            if not use_cache:  # template differs from what was cached: start clean
                from mlx_lm.models.cache import make_prompt_cache

                self.cache = make_prompt_cache(self.model)
                prompt, before = full, 0
            else:
                prompt, before = full[len(self.prefix) :], len(self.prefix)
            sampler = make_sampler(temp=self.temperature)  # 0: most likely words, repeatable
            t0 = time.perf_counter()
            last = None
            try:
                for resp in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,
                    sampler=sampler,
                    prompt_cache=self.cache,
                ):
                    answer.text += resp.text
                    last = resp
                    if on_text:
                        on_text(answer.text)
            finally:
                used = self.cache[0].offset - before
                if use_cache and used > 0:
                    trim_prompt_cache(self.cache, used)  # back to just the results
                elif not use_cache:
                    self.key = None  # rebuild the cached prefix next time
            answer.text = _clean(answer.text)
            answer.seconds = time.perf_counter() - t0
            if last is not None:
                answer.tokens_per_second = last.generation_tps
                answer.peak_memory_gb = last.peak_memory
            answer.done = True
        return answer


def _clean(text: str) -> str:
    """Drop any empty thinking block some models emit."""
    return re.sub(r"<think>\s*</think>\s*", "", text).strip()
