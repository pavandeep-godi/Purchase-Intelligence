"""Generate reproducible vendor offers and purchase history for the demo."""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

SEED = 47
QUARTERS = pd.period_range("2024Q1", "2026Q2", freq="Q")

MATERIALS = {
    "Methanol": ("Solvents", "Alcohols", 34.0),
    "Acetone": ("Solvents", "Ketones", 76.0),
    "Toluene": ("Solvents", "Aromatics", 91.0),
    "Ethyl Acetate": ("Solvents", "Esters", 104.0),
    "Caustic Soda": ("Inorganic Chemicals", "Alkalis", 39.0),
    "Hydrochloric Acid": ("Inorganic Chemicals", "Mineral Acids", 12.0),
    "Sulphuric Acid": ("Inorganic Chemicals", "Mineral Acids", 9.0),
    "Sodium Carbonate": ("Inorganic Chemicals", "Carbonates", 28.0),
    "Titanium Dioxide": ("Pigments", "Oxides", 245.0),
    "Carbon Black": ("Pigments", "Carbon Materials", 112.0),
    "Polyethylene Resin": ("Polymers", "Polyolefins", 126.0),
    "Polypropylene Resin": ("Polymers", "Polyolefins", 119.0),
    "Epoxy Resin": ("Polymers", "Thermosets", 228.0),
    "Citric Acid": ("Specialty Chemicals", "Organic Acids", 82.0),
    "Glycerine": ("Specialty Chemicals", "Polyols", 96.0),
    "Urea": ("Agrochemicals", "Nitrogen Compounds", 31.0),
}

VENDORS = [
    "Aarav Chemtrade", "Bharat Specialities", "Coastal Industrial Supply",
    "Deccan Molecules", "Eastern ChemSource", "Evergreen Materials",
    "Frontier Chemical Co", "Global Process Partners", "Horizon Petrochem",
    "Indus Industrial Solutions", "Jupiter Fine Chemicals", "Kaveri Trading",
    "Lotus Raw Materials", "Meridian ChemLink", "Nexus Bulk Supply",
    "Oceanic Ingredients", "Pioneer Chemical House", "Quartz Materials India",
    "Ridgeway Chemicals", "Summit Process Supply", "Trident ChemWorks",
    "Unity Industrial Partners", "Vantage Chemical Supply", "Westport Materials",
]

COUNTRIES = ["India", "China", "Singapore", "South Korea", "Japan", "Thailand", "UAE", "Germany"]


def _make_offers(rng: random.Random) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    vendor_bias = {vendor: rng.uniform(0.90, 1.12) for vendor in VENDORS}
    for material, (_, _, base_price) in MATERIALS.items():
        # Every material is offered by multiple vendors and in multiple origin countries.
        selected_vendors = rng.sample(VENDORS, k=rng.randint(8, 13))
        for vendor in selected_vendors:
            countries = rng.sample(COUNTRIES, k=rng.randint(1, 3))
            for country in countries:
                origin_factor = {
                    "India": 0.99, "China": 0.94, "Singapore": 1.04,
                    "South Korea": 1.08, "Japan": 1.12, "Thailand": 0.97,
                    "UAE": 1.03, "Germany": 1.16,
                }[country]
                price = base_price * vendor_bias[vendor] * origin_factor * rng.uniform(0.96, 1.045)
                freight = (0.0 if country == "India" else base_price * rng.uniform(0.018, 0.095))
                rows.append({
                    "vendor": vendor,
                    "material": material,
                    "country": country,
                    "price_per_kg_inr": round(price, 2),
                    "freight_cost_to_india_per_kg_inr": round(freight, 2),
                })
    return pd.DataFrame(rows).sort_values(["material", "country", "vendor"]).reset_index(drop=True)


def _make_purchases(offers: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offers_by_material: dict[str, list[dict[str, object]]] = {
        material: group.to_dict("records") for material, group in offers.groupby("material")
    }
    order_number = 1
    for quarter_index, quarter in enumerate(QUARTERS):
        # Gradual growth plus lumpy quarterly demand creates a realistic time series.
        line_count = 100 + quarter_index * 7
        for _ in range(line_count):
            material = rng.choice(list(MATERIALS))
            offer = rng.choice(offers_by_material[material])
            country = str(offer["country"])
            vendor = str(offer["vendor"])
            base_price = float(offer["price_per_kg_inr"])
            base_freight = float(offer["freight_cost_to_india_per_kg_inr"])
            demand_factor = 1.0 + quarter_index * 0.012
            quantity = rng.choice([500, 1000, 2000, 5000, 10000, 20000]) * demand_factor * rng.uniform(0.75, 1.3)
            seasonal = 1.0 + 0.025 * (quarter_index % 4)
            price = base_price * seasonal * rng.uniform(0.965, 1.07)
            freight = base_freight * rng.uniform(0.92, 1.12) if base_freight else 0.0
            # A small number of commercially valid but unusually expensive lines for anomaly detection.
            if rng.random() < 0.018:
                price *= rng.uniform(1.28, 1.65)
            if country != "India" and rng.random() < 0.012:
                freight *= rng.uniform(1.5, 2.2)
            quarter_start = quarter.start_time
            day_offset = rng.randint(0, 89)
            purchase_date = quarter_start + pd.Timedelta(days=day_offset)
            category, family, _ = MATERIALS[material]
            rows.append({
                "purchase_order_id": f"PO-{order_number:06d}",
                "purchase_date": purchase_date.strftime("%Y-%m-%d"),
                "quarter": str(quarter),
                "material": material,
                "quantity_kg": round(quantity, 2),
                "price_per_kg_inr": round(price, 2),
                "vendor": vendor,
                "source_country": country,
                "freight_cost_to_india_per_kg_inr": round(freight, 2),
                "chemical_category": category,
                "chemical_family": family,
            })
            order_number += 1
    return pd.DataFrame(rows)


def generate_mock_data(output_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Write the two requested CSV files and return their paths."""
    root = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "data"
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    offers = _make_offers(rng)
    purchases = _make_purchases(offers, rng)
    vendor_path = root / "vendor_offers.csv"
    purchase_path = root / "purchase_history.csv"
    offers.to_csv(vendor_path, index=False)
    purchases.to_csv(purchase_path, index=False)
    return vendor_path, purchase_path


if __name__ == "__main__":
    vendor_file, purchase_file = generate_mock_data()
    print(f"Created {vendor_file}")
    print(f"Created {purchase_file}")
