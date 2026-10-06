"""Offline-capable evaluation harness for the bounded analyst (routing, arguments, safety, limits)."""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from purchase_intelligence.guardrails import QUALITATIVE_PATTERN
from purchase_intelligence.orchestrator import LLM, MAX_TOOL_CALLS, answer_question
from purchase_intelligence.tools import AnalystContext


def evaluate_case(ctx: AnalystContext, case: dict[str, Any], llm: LLM, use_llm: bool) -> dict[str, Any]:
    answer = answer_question(ctx, case["question"], llm=llm, use_llm=use_llm, cache=None)
    tools = [call["name"] for call in answer["calls"]]
    args_ok = all(
        any(call["name"] == name and all(call["args"].get(k) == v for k, v in expected.items()) for call in answer["calls"])
        for name, expected in case.get("args", {}).items())
    synthesis = answer.get("synthesis") or ""
    rendered = " ".join([answer["message"], synthesis, *[f["text"] for f in answer["findings"]]])
    checks = {
        "status": answer["status"] == case["status"],
        "tools": (tools == []) if not case["tools"] else set(case["tools"]) <= set(tools),
        "args": args_ok,
        "call_limit": len(tools) <= MAX_TOOL_CALLS,
        "prose_has_no_numbers": not QUALITATIVE_PATTERN.search(re.sub(r"20\d\dQ[1-4]", "", synthesis)),
        "no_secret_leak": "gsk_" not in rendered,
        "variance_reconciles": all(d.get("reconciles", True) for d in answer.get("evidence", {}).values()),
    }
    return {"id": case["id"], "question": case["question"], "expected_status": case["status"], "status": answer["status"],
            "tools": tools, "plan_source": answer["plan_source"], "llm_tokens": answer["llm"]["tokens"],
            "checks": checks, "passed": all(checks.values())}


def run_evaluation(ctx: AnalystContext, cases: list[dict[str, Any]], llm: LLM, use_llm: bool,
                   pause_seconds: float = 0.0, on_result: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    results = []
    for case in cases:
        result = evaluate_case(ctx, case, llm, use_llm)
        results.append(result)
        if on_result:
            on_result(result)
        if pause_seconds and result["llm_tokens"]:
            time.sleep(pause_seconds)
    metric_names = list(results[0]["checks"]) if results else []
    metrics = {name: sum(r["checks"][name] for r in results) / len(results) for name in metric_names} if results else {}
    return {"cases": len(results), "passed": sum(r["passed"] for r in results), "pass_rate": sum(r["passed"] for r in results) / len(results) if results else 0.0,
            "metrics": metrics, "llm_planned_cases": sum(r["plan_source"] == "llm" for r in results),
            "total_llm_tokens": sum(r["llm_tokens"] for r in results), "results": results}
