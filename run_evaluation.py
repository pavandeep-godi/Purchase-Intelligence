"""Run the analyst evaluation set: python run_evaluation.py [--mode rules|live] [--pause 8]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from purchase_intelligence.evaluation import run_evaluation
from purchase_intelligence.llm import BUDGET, groq_json
from purchase_intelligence.tools import AnalystContext

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rules", "live"], default="rules",
                        help="rules: offline router, no Groq calls. live: real Groq planner and explanation (uses free-tier tokens).")
    parser.add_argument("--pause", type=float, default=14.0, help="seconds between live cases; about 1.2k tokens per case keeps usage under the 8,000 tokens/min limit")
    args = parser.parse_args()
    ctx = AnalystContext.build(pd.read_csv(ROOT / "data" / "purchase_history.csv"), pd.read_csv(ROOT / "data" / "vendor_offers.csv"))
    cases = json.loads((ROOT / "evaluation" / "cases.json").read_text(encoding="utf-8"))
    report = run_evaluation(
        ctx, cases, groq_json, use_llm=args.mode == "live", pause_seconds=args.pause if args.mode == "live" else 0.0,
        on_result=lambda r: print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<26} {r['status']:<11} tools={r['tools']} source={r['plan_source']}"
                                  + ("" if r["passed"] else f"  failed={[k for k, v in r['checks'].items() if not v]}")))
    print(f"\n{report['passed']}/{report['cases']} passed ({report['pass_rate']:.0%}); cases planned by Groq: {report['llm_planned_cases']}; LLM tokens used: {report['total_llm_tokens']}")
    for name, value in report["metrics"].items():
        print(f"  {name:<22} {value:.0%}")
    if args.mode == "live":
        print("Budget:", BUDGET.snapshot())
    out = ROOT / "reports" / f"evaluation_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
