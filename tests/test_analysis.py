"""Core validation tests for deterministic purchase analysis."""

from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import pandas as pd

from purchase_intelligence.agents.data_quality_agent import run_quality_checks
from purchase_intelligence.agents import groq_analysis_agent
from purchase_intelligence.agents.groq_analysis_agent import build_groq_context, run_groq_review
from purchase_intelligence.agents.spend_analysis_agent import analyze_spend

ROOT = Path(__file__).resolve().parents[1]


class PurchaseAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.purchases = pd.read_csv(ROOT / "data" / "purchase_history.csv")
        cls.offers = pd.read_csv(ROOT / "data" / "vendor_offers.csv")
        cls.analysis = analyze_spend(cls.purchases, cls.offers)

    def test_mock_data_covers_ten_quarters(self) -> None:
        self.assertEqual(len(self.purchases["quarter"].unique()), 10)
        self.assertGreater(len(self.purchases), 1000)
        self.assertGreaterEqual(self.offers["material"].nunique(), 12)

    def test_spend_reconciles_to_purchase_rows(self) -> None:
        summary = self.analysis["summary"]
        self.assertEqual(summary["valid_purchase_lines"], len(self.purchases))
        self.assertTrue(self.analysis["validations"]["landed_spend_equals_material_plus_freight"])
        self.assertAlmostEqual(summary["landed_spend_inr"], summary["material_spend_inr"] + summary["freight_spend_inr"], places=2)

    def test_savings_are_nonnegative_and_recomputed(self) -> None:
        savings = self.analysis["savings_lines"]
        self.assertTrue(self.analysis["validations"]["combined_savings_recomputed_from_unit_rates"])
        self.assertTrue((savings["combined_saving_inr"] >= 0).all())
        self.assertEqual(int(savings["has_comparable_offer"].sum()), len(savings))

    def test_invalid_purchase_row_is_flagged_and_excluded(self) -> None:
        broken = self.purchases.head(1).copy()
        broken.loc[broken.index[0], "quantity_kg"] = -5
        quality = run_quality_checks(broken, self.offers)
        analysis = analyze_spend(broken, self.offers)
        self.assertEqual(quality["invalid_purchase_rows"], 1)
        self.assertEqual(analysis["summary"]["excluded_purchase_rows"], 1)
        self.assertEqual(analysis["summary"]["landed_spend_inr"], 0)

    def test_duplicate_offers_are_reported_without_crashing(self) -> None:
        duplicate_offers = pd.concat([self.offers, self.offers.iloc[[0]]], ignore_index=True)
        quality = run_quality_checks(self.purchases.head(2), duplicate_offers)
        self.assertEqual(quality["duplicate_vendor_offers"], 1)

    def test_missing_offer_column_is_reported_without_crashing(self) -> None:
        malformed_offers = self.offers.drop(columns=["price_per_kg_inr"])
        quality = run_quality_checks(self.purchases.head(2), malformed_offers)
        analysis = analyze_spend(self.purchases.head(2), malformed_offers)
        self.assertIn("price_per_kg_inr", " ".join(quality["findings"]))
        self.assertGreater(analysis["summary"]["landed_spend_inr"], 0)
        self.assertEqual(analysis["summary"]["lines_with_comparable_offer"], 0)

    def test_groq_kpi_mismatch_uses_python_and_rejects_unsafe_summary(self) -> None:
        quality = run_quality_checks(self.purchases, self.offers)
        context = build_groq_context(self.analysis, quality, self.offers)
        metrics = dict(context["python_reconciliation"])
        metrics["landed_spend_inr"] += 1000
        payload = {
            "dashboard_metrics": metrics,
            "executive_summary": "The data is clean and fully reconciled.",
            "calculation_checks": [],
            "additional_validation_findings": [],
            "dashboard_insights": [],
        }
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
        with patch("purchase_intelligence.agents.groq_analysis_agent._get_client", return_value=(client, "test client")):
            review = run_groq_review(context)
        self.assertFalse(review["metric_audit"]["landed_spend_inr"]["matched"])
        self.assertEqual(review["dashboard_metrics"]["landed_spend_inr"], self.analysis["summary"]["landed_spend_inr"])
        self.assertFalse(review["executive_summary_validated"])
        self.assertIn("valid purchase lines", review["executive_summary"])

    def test_groq_key_loads_from_streamlit_cloud_secrets(self) -> None:
        cloud_secrets = SimpleNamespace(secrets={"GROQ_API_KEY": "test-groq-key"})
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(groq_analysis_agent, "ROOT", Path(temporary_directory)),
                patch.dict(os.environ, {}, clear=True),
                patch.dict(sys.modules, {"streamlit": cloud_secrets}),
            ):
                groq_analysis_agent.load_groq_environment()
                self.assertEqual(os.environ["GROQ_API_KEY"], "test-groq-key")


if __name__ == "__main__":
    unittest.main()
