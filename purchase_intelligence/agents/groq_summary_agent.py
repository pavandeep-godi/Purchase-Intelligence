"""Optional Groq narrative generation; all metric calculations remain in Python."""

from __future__ import annotations

import json
import os

from purchase_intelligence.agents.groq_analysis_agent import load_groq_environment


def create_narrative(summary: dict[str, object], findings: list[str]) -> tuple[str, str]:
    """Ask Groq for a short plain-English summary, or use the offline fallback."""
    load_groq_environment()
    fallback = (
        f"The analysis covers {summary['valid_purchase_lines']:,} valid purchase lines across "
        f"{summary['vendor_count']} vendors and {summary['material_count']} materials. "
        f"Total landed spend is ₹{summary['landed_spend_inr']:,.0f}; indicative comparable-offer "
        f"savings are ₹{summary['combined_addressable_savings_inr']:,.0f} "
        f"({summary['savings_as_pct_of_landed_spend']:.1f}% of landed spend). "
        f"Comparable quote coverage is {summary['comparable_offer_coverage_pct']:.1f}%. "
        "Savings are directional and require commercial, quality, and capacity review."
    )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return fallback, "Local deterministic summary (GROQ_API_KEY not set)"
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            temperature=0.2,
            max_tokens=500,
            messages=[
                {"role": "system", "content": (
                    "You are a procurement analyst at a chemicals manufacturer. Write a concise, "
                    "plain-English executive summary for non-technical business users. Use only "
                    "the supplied validated metrics and findings. Do not recalculate, invent, or "
                    "overstate savings. Explicitly say opportunities are indicative and require validation."
                )},
                {"role": "user", "content": json.dumps({"validated_metrics": summary, "data_quality_findings": findings}, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip(), "Groq narrative (metrics computed and validated in Python)"
        return fallback, "Local summary (Groq returned no text)"
    except Exception as exc:  # API/network/model failures must not break the dashboard.
        return fallback, f"Local deterministic summary (Groq unavailable: {type(exc).__name__})"
