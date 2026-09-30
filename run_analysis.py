"""Generate the text and JSON purchase summary reports."""

from pathlib import Path

from purchase_intelligence.agents.workflow import run_workflow

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    result = run_workflow(ROOT / "data" / "purchase_history.csv", ROOT / "data" / "vendor_offers.csv", ROOT / "reports")
    print("Analysis complete")
    print(f"Landed spend: ₹{result['summary']['landed_spend_inr']:,.0f}")
    print(f"Indicative combined savings: ₹{result['summary']['combined_addressable_savings_inr']:,.0f}")
    print(f"Groq analysis: {result['llm_review']['status']}")
    print(f"Report saved to: {ROOT / 'reports' / 'purchase_summary.txt'}")
