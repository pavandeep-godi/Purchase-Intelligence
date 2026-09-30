"""Deterministic spend, supplier, and savings calculations."""

from __future__ import annotations

import pandas as pd


def prepare_purchase_lines(purchases: pd.DataFrame) -> pd.DataFrame:
    """Normalize purchase records and calculate line-level cost fields."""
    data = purchases.copy()
    numeric_columns = ["quantity_kg", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"]
    for column in numeric_columns:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["vendor", "material", "source_country", "chemical_category", "chemical_family", "quarter"]:
        if column not in data:
            data[column] = "Unknown"
        data[column] = data[column].fillna("Unknown").astype(str)
    valid = (
        data["quantity_kg"].gt(0) & data["price_per_kg_inr"].ge(0)
        & data["freight_cost_to_india_per_kg_inr"].ge(0)
        & data[numeric_columns].notna().all(axis=1)
    )
    data = data.loc[valid].copy()
    data["material_spend_inr"] = data["quantity_kg"] * data["price_per_kg_inr"]
    data["freight_spend_inr"] = data["quantity_kg"] * data["freight_cost_to_india_per_kg_inr"]
    data["landed_cost_per_kg_inr"] = data["price_per_kg_inr"] + data["freight_cost_to_india_per_kg_inr"]
    data["landed_spend_inr"] = data["material_spend_inr"] + data["freight_spend_inr"]
    if "purchase_date" in data:
        data["purchase_date"] = pd.to_datetime(data["purchase_date"], errors="coerce")
    return data


def calculate_savings(lines: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Compare each line against best same-material, same-country vendor offers."""
    data = lines.copy()
    quote_data = offers.copy()
    for column in ["vendor", "material", "country", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"]:
        if column not in quote_data:
            quote_data[column] = pd.NA
    quote_data["price_per_kg_inr"] = pd.to_numeric(quote_data["price_per_kg_inr"], errors="coerce")
    quote_data["freight_cost_to_india_per_kg_inr"] = pd.to_numeric(quote_data["freight_cost_to_india_per_kg_inr"], errors="coerce")
    quote_data = quote_data.dropna(subset=["material", "country", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"])
    quote_data = quote_data[(quote_data["price_per_kg_inr"] >= 0) & (quote_data["freight_cost_to_india_per_kg_inr"] >= 0)]
    quote_data["offer_landed_per_kg_inr"] = quote_data["price_per_kg_inr"] + quote_data["freight_cost_to_india_per_kg_inr"]
    best_price = quote_data.sort_values(["price_per_kg_inr", "vendor"]).drop_duplicates(["material", "country"])[
        ["material", "country", "price_per_kg_inr", "vendor"]
    ].rename(columns={"price_per_kg_inr": "benchmark_price_per_kg_inr", "vendor": "best_price_vendor"})
    best_freight = quote_data.sort_values(["freight_cost_to_india_per_kg_inr", "vendor"]).drop_duplicates(["material", "country"])[
        ["material", "country", "freight_cost_to_india_per_kg_inr", "vendor"]
    ].rename(columns={"freight_cost_to_india_per_kg_inr": "benchmark_freight_per_kg_inr", "vendor": "best_freight_vendor"})
    best_landed = quote_data.sort_values(["offer_landed_per_kg_inr", "vendor"]).drop_duplicates(["material", "country"])[
        ["material", "country", "offer_landed_per_kg_inr", "vendor"]
    ].rename(columns={"offer_landed_per_kg_inr": "benchmark_landed_per_kg_inr", "vendor": "best_landed_vendor"})
    result = data
    for benchmark in [best_price, best_freight, best_landed]:
        result = result.merge(benchmark, left_on=["material", "source_country"], right_on=["material", "country"], how="left")
        result = result.drop(columns=["country"], errors="ignore")
    result["price_saving_per_kg_inr"] = (result["price_per_kg_inr"] - result["benchmark_price_per_kg_inr"]).clip(lower=0).fillna(0)
    result["freight_saving_per_kg_inr"] = (result["freight_cost_to_india_per_kg_inr"] - result["benchmark_freight_per_kg_inr"]).clip(lower=0).fillna(0)
    result["combined_saving_per_kg_inr"] = (result["landed_cost_per_kg_inr"] - result["benchmark_landed_per_kg_inr"]).clip(lower=0).fillna(0)
    result["price_saving_inr"] = result["price_saving_per_kg_inr"] * result["quantity_kg"]
    result["freight_saving_inr"] = result["freight_saving_per_kg_inr"] * result["quantity_kg"]
    result["combined_saving_inr"] = result["combined_saving_per_kg_inr"] * result["quantity_kg"]
    result["has_comparable_offer"] = result["benchmark_landed_per_kg_inr"].notna()
    return result


def analyze_spend(purchases: pd.DataFrame, offers: pd.DataFrame) -> dict[str, object]:
    """Return line-level analysis, summaries, and explicit arithmetic validations."""
    lines = prepare_purchase_lines(purchases)
    savings = calculate_savings(lines, offers)
    quantity = float(lines["quantity_kg"].sum()) if not lines.empty else 0.0
    material_spend = float(lines["material_spend_inr"].sum()) if not lines.empty else 0.0
    freight_spend = float(lines["freight_spend_inr"].sum()) if not lines.empty else 0.0
    landed_spend = float(lines["landed_spend_inr"].sum()) if not lines.empty else 0.0
    combined_savings = float(savings["combined_saving_inr"].sum()) if not savings.empty else 0.0
    savings_recomputed = float((savings["combined_saving_per_kg_inr"] * savings["quantity_kg"]).sum()) if not savings.empty else 0.0
    summary = {
        "purchase_rows_in_file": int(len(purchases)), "valid_purchase_lines": int(len(lines)),
        "excluded_purchase_rows": int(len(purchases) - len(lines)),
        "vendor_count": int(lines["vendor"].nunique()) if not lines.empty else 0,
        "material_count": int(lines["material"].nunique()) if not lines.empty else 0,
        "total_quantity_kg": quantity, "material_spend_inr": material_spend,
        "freight_spend_inr": freight_spend, "landed_spend_inr": landed_spend,
        "freight_share_pct": freight_spend / landed_spend * 100 if landed_spend else 0.0,
        "average_landed_cost_per_kg_inr": landed_spend / quantity if quantity else 0.0,
        "combined_addressable_savings_inr": combined_savings,
        "savings_as_pct_of_landed_spend": combined_savings / landed_spend * 100 if landed_spend else 0.0,
        "lines_with_comparable_offer": int(savings["has_comparable_offer"].sum()) if not savings.empty else 0,
        "comparable_offer_coverage_pct": float(savings["has_comparable_offer"].mean() * 100) if not savings.empty else 0.0,
    }
    validations = {
        "landed_spend_equals_material_plus_freight": abs(landed_spend - material_spend - freight_spend) < 0.01,
        "landed_spend_reconciliation_delta_inr": landed_spend - material_spend - freight_spend,
        "combined_savings_recomputed_from_unit_rates": abs(combined_savings - savings_recomputed) < 0.01,
        "combined_savings_recalculation_delta_inr": combined_savings - savings_recomputed,
        "valid_lines_do_not_exceed_file_rows": len(lines) <= len(purchases),
    }
    quarterly = lines.groupby("quarter", as_index=False).agg(
        material_spend_inr=("material_spend_inr", "sum"), freight_spend_inr=("freight_spend_inr", "sum"),
        landed_spend_inr=("landed_spend_inr", "sum"), quantity_kg=("quantity_kg", "sum"),
        purchase_lines=("quantity_kg", "size"),
    ) if not lines.empty else pd.DataFrame()
    if not quarterly.empty:
        quarterly["average_landed_cost_per_kg_inr"] = quarterly["landed_spend_inr"] / quarterly["quantity_kg"].replace(0, pd.NA)
        quarterly = quarterly.sort_values("quarter")
    vendors = savings.groupby("vendor", as_index=False).agg(
        landed_spend_inr=("landed_spend_inr", "sum"), material_spend_inr=("material_spend_inr", "sum"),
        freight_spend_inr=("freight_spend_inr", "sum"), quantity_kg=("quantity_kg", "sum"),
        purchase_lines=("quantity_kg", "size"), combined_savings_inr=("combined_saving_inr", "sum"),
        price_savings_inr=("price_saving_inr", "sum"), freight_savings_inr=("freight_saving_inr", "sum"),
    ) if not savings.empty else pd.DataFrame()
    if not vendors.empty:
        vendors["average_landed_cost_per_kg_inr"] = vendors["landed_spend_inr"] / vendors["quantity_kg"].replace(0, pd.NA)
        vendors["spend_share_pct"] = vendors["landed_spend_inr"] / landed_spend * 100 if landed_spend else 0.0
    categories = savings.groupby("chemical_category", as_index=False).agg(
        landed_spend_inr=("landed_spend_inr", "sum"), quantity_kg=("quantity_kg", "sum"),
        combined_savings_inr=("combined_saving_inr", "sum"),
    ) if not savings.empty else pd.DataFrame()
    return {"lines": lines, "savings_lines": savings, "summary": summary, "validations": validations,
            "quarterly": quarterly, "vendors": vendors, "categories": categories}
