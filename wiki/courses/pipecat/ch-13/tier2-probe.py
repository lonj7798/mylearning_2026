#!/usr/bin/env python3
"""tier2-probe.py — the ONE measurement the ch-13 capstone actually runs.

WHAT THIS MEASURES. Wall-clock latency of boson's Tier-2 rule evaluation: the two rules
stamped ``@check(..., check_type="llm")`` in Lina's ``03-orchestrator`` layer —
``intent_rules`` (priority 30) and ``sentiment_tracker`` (priority 10). Both are
``mode="parallel"``, so ``RuleEngine.evaluate`` runs them inside ONE ``asyncio.gather``
and a turn pays max(), not sum(). This script reproduces that shape — per user turn, one
``asyncio.gather`` of two chat completions against your own endpoint — and prints P50/P95
of per-turn wall clock in milliseconds.

WHY THIS IS THE ONE MEASUREMENT. ch-12 billed Tier-2 at "~250-400 ms of added pre-LLM
latency on every turn": an ESTIMATE, not an observation. Row 13 of the vote table and the
ch-11 budget both hang off it. Of the four numbers this course proved nobody has, three
need a telephony carrier, a Korean STT contract, or 8 kHz audio. This one needs none of
them — two text prompts against an endpoint you already run — so it is the only one that
can be RUN rather than listed, and running it turns row 13 from pending into decided.
``figures/migration-map.html`` is an offline ``file://`` page and cannot call an endpoint;
that is why the measurement lives here and the figure takes the two printed numbers as
pasted input.

USAGE (TIER2_API_KEY optional; argv order is ENDPOINT MODEL ITERATIONS)
    export TIER2_ENDPOINT="http://localhost:8000/v1/chat/completions"; export TIER2_MODEL="..."
    python3 tier2-probe.py --iterations 40
"""

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Real Lina TMR user turns. Override with --corpus <file> (one turn per line).
CORPUS = [
    "네 말씀하세요",
    "그거 지금 얼마짜리예요?",
    "아 지금은 좀 바쁜데요",
    "보험료가 매달 얼마나 나가는 건가요",
    "가족이랑 상의해보고 다시 연락드릴게요",
    "네 동의합니다 진행해주세요",
    "다시는 전화하지 마세요",
]

INTENT_SYSTEM = (
    "You classify a telemarketing turn against a list of intent descriptions. Most recent "
    "turn (PRIMARY SIGNAL — evaluate against THESE). Answer with a comma-separated list of "
    "matching indices, or the word none. No prose.\n0: Agent has explicitly asked the "
    "customer for permission to continue.\n1: Customer yields the floor (direct consent)."
)
SENTIMENT_SYSTEM = (
    "You track customer sentiment across a telemarketing call. Answer with exactly one "
    "word: positive, neutral, or negative. No prose."
)


def _post(endpoint, api_key, model, system, user, timeout):
    """One blocking chat completion. urllib keeps this stdlib-only."""
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 16,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


async def one_turn(endpoint, api_key, model, turn, timeout):
    """Wall clock of the parallel phase for one finished utterance, in ms."""
    started = time.perf_counter()
    await asyncio.gather(
        asyncio.to_thread(_post, endpoint, api_key, model, INTENT_SYSTEM, turn, timeout),
        asyncio.to_thread(_post, endpoint, api_key, model, SENTIMENT_SYSTEM, turn, timeout),
    )
    return (time.perf_counter() - started) * 1000.0


def percentile(samples, q):
    """Nearest-rank percentile — honest for the small N this probe produces."""
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), -(-len(ordered) * q // 100)))
    return ordered[int(rank) - 1]


async def run(args):
    corpus = CORPUS
    if args.corpus:
        with open(args.corpus, encoding="utf-8") as fh:
            corpus = [line.strip() for line in fh if line.strip()]
    if not corpus:
        print("tier2-probe: corpus is empty; nothing to measure.", file=sys.stderr)
        return 2

    samples = []
    for i in range(args.iterations):
        turn = corpus[i % len(corpus)]
        try:
            ms = await one_turn(args.endpoint, args.api_key, args.model, turn, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            print(
                f"tier2-probe: UNMEASURED. The endpoint did not answer.\n"
                f"  endpoint : {args.endpoint}\n  model    : {args.model}\n"
                f"  failed on: iteration {i + 1}/{args.iterations}, turn {turn!r}\n"
                f"  reason   : {type(exc).__name__}: {exc}\n"
                "Bring the endpoint up, or point TIER2_ENDPOINT at a reachable one, and\n"
                "re-run. Do NOT paste an estimate into figures/migration-map.html — row 13\n"
                "stays pending until this prints two real numbers.",
                file=sys.stderr,
            )
            return 1
        samples.append(ms)
        if args.verbose:
            print(f"  turn {i + 1:>3}: {ms:8.1f} ms  {turn}", file=sys.stderr)

    print(f"tier2-probe: {len(samples)} turns against {args.model} at {args.endpoint}"
          f"   [min {min(samples):.1f} / max {max(samples):.1f} ms]\n"
          f"  P50  {percentile(samples, 50):8.1f} ms\n  P95  {percentile(samples, 95):8.1f} ms")
    print("Paste P50 and P95 into the MEASUREMENT panel of figures/migration-map.html.")
    return 0


def main():
    p = argparse.ArgumentParser(description="Measure boson Tier-2 rule-evaluation latency.")
    p.add_argument("endpoint", nargs="?", default=os.environ.get("TIER2_ENDPOINT", ""))
    p.add_argument("model", nargs="?", default=os.environ.get("TIER2_MODEL", "Qwen3.6-27B-FP8"))
    p.add_argument("iterations", nargs="?", type=int, default=int(os.environ.get("TIER2_ITERATIONS", "40")))
    p.add_argument("--corpus", default=os.environ.get("TIER2_CORPUS", ""))
    p.add_argument("--timeout", type=float, default=float(os.environ.get("TIER2_TIMEOUT", "10")))
    p.add_argument("--api-key", default=os.environ.get("TIER2_API_KEY", ""))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if not args.endpoint:
        print("tier2-probe: no endpoint. Set TIER2_ENDPOINT or pass it as argv[1]:\n  python3 "
              "tier2-probe.py http://localhost:8000/v1/chat/completions Qwen3.6-27B-FP8 40",
              file=sys.stderr)
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
