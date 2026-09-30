"""Orchestrate data checks, deterministic calculations, and text reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from purchase_intelligence.agents.data_quality_agent import run_quality_checks
from purchase_intelligence.agents.groq_analysis_agent import build_groq_context, run_groq_review
from purchase_intelligence.agents.spend_analysis_agent import analyze_spend


def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"


def run_workflow(purchase_file: Path | str, vendor_file: Path | str, output_dir: Path | str | None = None) -> dict[str, object]:
    """Read the two CSVs, analyze, and optionally persist the requested summary text."""
    purchases = pd.read_csv(purchase_file)
    offers = pd.read_csv(vendor_file)
    quality = run_quality_checks(purchases, offers)
    analysis = analyze_spend(purchases, offers)
    groq_context = build_groq_context(analysis, quality, offers)
    llm_review = run_groq_review(groq_context)
    narrative = llm_review["executive_summary"]
    result = {**analysis, "quality": quality, "narrative": narrative, "llm_review": llm_review}
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary = analysis["summary"]
        validations = analysis["validations"]
        lines = [
            "PURCHASE INTELLIGENCE — PURCHASE & VENDOR SPEND SUMMARY",
            "=" * 62,
            "",
            "EXECUTIVE SUMMARY",
            narrative,
            f"Analysis source: {llm_review['source']}",
            "",
            "KEY METRICS (Groq-calculated where reconciled; otherwise Python fallback)",
            f"Purchase rows in source file: {summary['purchase_rows_in_file']:,}",
            f"Valid purchase lines used: {summary['valid_purchase_lines']:,}",
            f"Excluded invalid purchase lines: {summary['excluded_purchase_rows']:,}",
            f"Vendors / materials: {summary['vendor_count']} / {summary['material_count']}",
            f"Quantity purchased: {summary['total_quantity_kg']:,.0f} kg",
            f"Material spend: {format_inr(summary['material_spend_inr'])}",
            f"Freight spend: {format_inr(summary['freight_spend_inr'])}",
            f"Landed spend (material + freight): {format_inr(summary['landed_spend_inr'])}",
            f"Freight share of landed spend: {summary['freight_share_pct']:.2f}%",
            f"Average landed cost: {format_inr(summary['average_landed_cost_per_kg_inr'])}/kg",
            f"Indicative combined offer savings: {format_inr(summary['combined_addressable_savings_inr'])}",
            f"Savings / landed spend: {summary['savings_as_pct_of_landed_spend']:.2f}%",
            f"Comparable quote coverage: {summary['comparable_offer_coverage_pct']:.2f}%",
            "",
            "DATA QUALITY & ANOMALIES",
            *[f"- {finding}" for finding in quality["findings"]],
                        "",
                        "GROQ VALIDATION & DASHBOARD METRIC REVIEW",
                        f"Status: {llm_review['status']}",
                        f"KPI values reconciled: {llm_review['matched_metric_count']}/{llm_review['metric_count']}",
                        f"Chart series source: {llm_review.get('dashboard_series_source', 'Python row-level aggregation')}.",
                        f"Executive narrative contains no unverified numeric claims: {'PASS' if llm_review.get('executive_summary_validated', False) else 'Python-generated fallback'}.",
                        "Additional LLM findings (review before action):",
                        *([f"- {finding}" for finding in llm_review["additional_validation_findings"]] or ["- None returned."]),
                        "LLM calculation checks:",
                        *([f"- {check.get('name', 'Check')}: {check.get('status', 'review')} — {check.get('explanation', '')}"
                             for check in llm_review.get("calculation_checks", []) if isinstance(check, dict)] or ["- None returned."]),
                        "Metric reconciliation detail:",
                        *[f"- {key}: {'PASS' if audit['matched'] else 'Python fallback'} (Groq={audit['llm_value']}, Python={audit['python_value']})"
                            for key, audit in llm_review["metric_audit"].items()],
                        "Dashboard insights:",
                        *([f"- {insight}" for insight in llm_review["dashboard_insights"]] or ["- None returned."]),
            "",
            "ARITHMETIC VALIDATION",
            f"Landed spend = material + freight: {'PASS' if validations['landed_spend_equals_material_plus_freight'] else 'FAIL'} "
            f"(difference {format_inr(validations['landed_spend_reconciliation_delta_inr'])})",
            f"Savings recompute from unit rate × kg: {'PASS' if validations['combined_savings_recomputed_from_unit_rates'] else 'FAIL'} "
            f"(difference {format_inr(validations['combined_savings_recalculation_delta_inr'])})",
            f"Valid line count ≤ source row count: {'PASS' if validations['valid_lines_do_not_exceed_file_rows'] else 'FAIL'}",
            "",
            "SAVINGS METHOD & LIMITATIONS",
            "Price, freight, and combined landed-cost savings compare each purchase only with quoted offers for the same material and source country. The combined opportunity uses the lowest quoted landed cost, so it is not the sum of independently best price and freight opportunities. Estimates assume comparable supply is available and exclude switching, qualification, tax/duty, FX, quality, capacity, lead-time, and contract costs. Review quotes before action.",
            "",
        ]
        (output_path / "purchase_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        json_result = {
            "summary": summary,
            "validations": validations,
            "quality": quality,
            "narrative": narrative,
            "llm_review": llm_review,
        }
        (output_path / "purchase_summary.json").write_text(json.dumps(json_result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
