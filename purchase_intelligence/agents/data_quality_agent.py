"""Data quality and anomaly checks for vendor offers and purchase lines."""

from __future__ import annotations

import pandas as pd

PURCHASE_REQUIRED = {
    "purchase_order_id", "purchase_date", "quarter", "material", "quantity_kg",
    "price_per_kg_inr", "vendor", "source_country",
    "freight_cost_to_india_per_kg_inr", "chemical_category", "chemical_family",
}
OFFER_REQUIRED = {
    "vendor", "material", "country", "price_per_kg_inr",
    "freight_cost_to_india_per_kg_inr",
}


def run_quality_checks(purchases: pd.DataFrame, offers: pd.DataFrame) -> dict[str, object]:
    """Return quality counts and concise, actionable findings."""
    findings: list[str] = []
    purchase_missing_columns = sorted(PURCHASE_REQUIRED - set(purchases.columns))
    offer_missing_columns = sorted(OFFER_REQUIRED - set(offers.columns))
    if purchase_missing_columns:
        findings.append(f"Purchase file is missing required columns: {', '.join(purchase_missing_columns)}.")
    if offer_missing_columns:
        findings.append(f"Vendor file is missing required columns: {', '.join(offer_missing_columns)}.")

    required_purchase_columns = sorted(PURCHASE_REQUIRED & set(purchases.columns))
    required_offer_columns = sorted(OFFER_REQUIRED & set(offers.columns))
    missing_purchase_cells = int(purchases[required_purchase_columns].isna().sum().sum()) if required_purchase_columns else 0
    missing_offer_cells = int(offers[required_offer_columns].isna().sum().sum()) if required_offer_columns else 0
    if missing_purchase_cells:
        findings.append(f"{missing_purchase_cells:,} required purchase cells are blank or missing.")
    if missing_offer_cells:
        findings.append(f"{missing_offer_cells:,} required vendor-offer cells are blank or missing.")

    invalid_offer_count = 0
    if {"price_per_kg_inr", "freight_cost_to_india_per_kg_inr"}.issubset(offers.columns):
        offer_rates = offers[["price_per_kg_inr", "freight_cost_to_india_per_kg_inr"]].apply(pd.to_numeric, errors="coerce")
        invalid_offer_count = int(
            offer_rates.isna().any(axis=1).sum()
            + ((offer_rates < 0).any(axis=1) & ~offer_rates.isna().any(axis=1)).sum()
        )
        if invalid_offer_count:
            findings.append(f"{invalid_offer_count:,} vendor offers have non-numeric or negative price/freight and should not be benchmarked.")

    duplicate_orders = int(purchases["purchase_order_id"].duplicated().sum()) if "purchase_order_id" in purchases else 0
    duplicate_offers = int(offers.duplicated(subset=["vendor", "material", "country"]).sum()) if {"vendor", "material", "country"}.issubset(offers.columns) else 0
    if duplicate_orders:
        findings.append(f"{duplicate_orders:,} duplicate purchase order IDs found; review before treating rows as unique orders.")
    if duplicate_offers:
        findings.append(f"{duplicate_offers:,} repeated vendor/material/country offer combinations found.")

    valid = purchases.copy()
    numeric = ["quantity_kg", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"]
    for column in numeric:
        if column in valid:
            valid[column] = pd.to_numeric(valid[column], errors="coerce")
    if set(numeric).issubset(valid.columns):
        invalid_mask = (
            valid[numeric].isna().any(axis=1)
            | (valid["quantity_kg"] <= 0)
            | (valid["price_per_kg_inr"] < 0)
            | (valid["freight_cost_to_india_per_kg_inr"] < 0)
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            findings.append(f"{invalid_count:,} purchase rows have non-numeric, zero/negative quantity, or negative cost and are excluded from spend metrics.")
    else:
        invalid_count = len(valid)

    invalid_date_count = 0
    if "purchase_date" in purchases:
        invalid_date_count = int(pd.to_datetime(purchases["purchase_date"], errors="coerce").isna().sum())
        if invalid_date_count:
            findings.append(f"{invalid_date_count:,} purchase rows have missing or unrecognizable purchase dates.")

    unmatched_count = 0
    offer_deviation_count = 0
    if (
        {"vendor", "material", "source_country", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"}.issubset(purchases.columns)
        and {"vendor", "material", "country", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"}.issubset(offers.columns)
    ):
        purchase_keys = purchases[["vendor", "material", "source_country"]].astype("string").fillna("")
        offer_keys = offers[["vendor", "material", "country"]].astype("string").fillna("")
        lookup = set(map(tuple, offer_keys.to_numpy()))
        unmatched = ~purchase_keys.apply(tuple, axis=1).isin(lookup)
        unmatched_count = int(unmatched.sum())
        if unmatched_count:
            findings.append(f"{unmatched_count:,} purchase lines have no matching vendor/material/source-country quote.")

        unique_offers = offers.drop_duplicates(subset=["vendor", "material", "country"], keep="last").copy()
        unique_offers["price_per_kg_inr"] = pd.to_numeric(unique_offers["price_per_kg_inr"], errors="coerce")
        unique_offers["freight_cost_to_india_per_kg_inr"] = pd.to_numeric(unique_offers["freight_cost_to_india_per_kg_inr"], errors="coerce")
        rates = purchases.merge(
            unique_offers.rename(columns={"country": "source_country", "price_per_kg_inr": "quoted_price_per_kg_inr", "freight_cost_to_india_per_kg_inr": "quoted_freight_per_kg_inr"}),
            on=["vendor", "material", "source_country"], how="left", validate="many_to_one",
        )
        purchase_prices = pd.to_numeric(rates["price_per_kg_inr"], errors="coerce")
        quoted_prices = pd.to_numeric(rates["quoted_price_per_kg_inr"], errors="coerce")
        purchase_freight = pd.to_numeric(rates["freight_cost_to_india_per_kg_inr"], errors="coerce")
        quoted_freight = pd.to_numeric(rates["quoted_freight_per_kg_inr"], errors="coerce")
        price_ratio = purchase_prices / quoted_prices.replace(0, pd.NA)
        freight_ratio = purchase_freight / quoted_freight.replace(0, pd.NA)
        price_flags = price_ratio.gt(1.25) | price_ratio.lt(0.75)
        freight_flags = freight_ratio.gt(1.50)
        offer_deviation_count = int((price_flags | freight_flags).fillna(False).sum())
        if offer_deviation_count:
            findings.append(f"{offer_deviation_count:,} lines have price more than 25% from quote or freight more than 50% above quote; review as potential leakage or stale quotes.")

    numeric_prices = pd.to_numeric(purchases.get("price_per_kg_inr", pd.Series(dtype=float)), errors="coerce")
    if "material" in purchases:
        material_medians = numeric_prices.groupby(purchases["material"]).transform("median")
        extreme_price_count = int((numeric_prices.gt(material_medians * 1.5) & material_medians.gt(0)).sum())
    else:
        extreme_price_count = 0
    if extreme_price_count:
        findings.append(f"{extreme_price_count:,} lines exceed 1.5× their material's median price; verify for unusual rates.")

    classification_conflicts = 0
    if {"material", "chemical_category", "chemical_family"}.issubset(purchases.columns):
        classifications = purchases.groupby("material")[["chemical_category", "chemical_family"]].nunique(dropna=True)
        classification_conflicts = int(classifications.gt(1).any(axis=1).sum())
        if classification_conflicts:
            findings.append(f"{classification_conflicts:,} materials have conflicting category/family classifications across purchase rows.")

    if not findings:
        findings.append("No missing fields, invalid costs, duplicates, unmatched offers, or material quote deviations were detected.")
    return {
        "findings": findings,
        "purchase_rows": int(len(purchases)),
        "vendor_offer_rows": int(len(offers)),
        "missing_purchase_cells": missing_purchase_cells,
        "missing_offer_cells": missing_offer_cells,
        "duplicate_purchase_order_ids": duplicate_orders,
        "duplicate_vendor_offers": duplicate_offers,
        "invalid_purchase_rows": invalid_count,
        "invalid_purchase_dates": invalid_date_count,
        "invalid_vendor_offers": invalid_offer_count,
        "unmatched_purchase_rows": unmatched_count,
        "offer_deviation_rows": offer_deviation_count,
        "extreme_price_rows": extreme_price_count,
        "classification_conflict_materials": classification_conflicts,
    }
