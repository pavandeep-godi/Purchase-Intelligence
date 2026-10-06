"""Approved, typed analysis tools. Every number comes from deterministic pandas code, never from an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from purchase_intelligence.agents.data_quality_agent import PURCHASE_REQUIRED, run_quality_checks
from purchase_intelligence.agents.spend_analysis_agent import analyze_spend, prepare_purchase_lines

# Same thresholds as the data-quality agent so "unreliable" means one thing everywhere.
PRICE_DEVIATION_HIGH, PRICE_DEVIATION_LOW, FREIGHT_DEVIATION_HIGH = 1.25, 0.75, 1.5
GROUP_COLUMNS = {"material": "material", "vendor": "vendor", "category": "chemical_category"}


class ToolArgError(ValueError):
    """Raised when a requested tool call has unknown tools or invalid arguments."""


@dataclass
class AnalystContext:
    purchases: pd.DataFrame
    offers: pd.DataFrame
    selected_quarter: str | None
    quarters: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    fingerprint: str = ""

    @classmethod
    def build(cls, purchases: pd.DataFrame, offers: pd.DataFrame, selected_quarter: str | None = None) -> "AnalystContext":
        quarters = sorted(purchases["quarter"].dropna().astype(str).unique()) if "quarter" in purchases else []
        materials = sorted(set(offers["material"].dropna().astype(str)) | set(purchases["material"].dropna().astype(str))) \
            if "material" in offers and "material" in purchases else []
        countries = sorted(offers["country"].dropna().astype(str).unique()) if "country" in offers else []
        digest = int(pd.util.hash_pandas_object(purchases, index=False).sum() % (1 << 32)) ^ \
            int(pd.util.hash_pandas_object(offers, index=False).sum() % (1 << 32))
        selected = selected_quarter if selected_quarter in quarters else (quarters[-1] if quarters else None)
        return cls(purchases, offers, selected, quarters, materials, countries, f"{digest:08x}")


@dataclass
class ToolResult:
    name: str
    status: str  # ok | empty | error
    args: dict[str, Any]
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def _quarter_frame(ctx: AnalystContext, quarter: str | None) -> pd.DataFrame:
    if quarter and "quarter" in ctx.purchases:
        return ctx.purchases[ctx.purchases["quarter"].astype(str).eq(quarter)].copy()
    return ctx.purchases.copy()


def _flag_lines(lines: pd.DataFrame, offers: pd.DataFrame) -> pd.Series:
    """True where a purchase has no own-vendor quote or deviates materially from it."""
    needed = {"vendor", "material", "country", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"}
    if not needed.issubset(offers.columns):
        return pd.Series(True, index=lines.index)
    quotes = offers.drop_duplicates(["vendor", "material", "country"], keep="last").rename(columns={
        "country": "source_country", "price_per_kg_inr": "q_price", "freight_cost_to_india_per_kg_inr": "q_freight"})
    merged = lines[["vendor", "material", "source_country", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"]].reset_index() \
        .merge(quotes[["vendor", "material", "source_country", "q_price", "q_freight"]],
               on=["vendor", "material", "source_country"], how="left").set_index("index")
    q_price = pd.to_numeric(merged["q_price"], errors="coerce")
    q_freight = pd.to_numeric(merged["q_freight"], errors="coerce")
    price_ratio = merged["price_per_kg_inr"] / q_price.where(q_price.ne(0))
    freight_ratio = merged["freight_cost_to_india_per_kg_inr"] / q_freight.where(q_freight.ne(0))
    flagged = q_price.isna() | price_ratio.gt(PRICE_DEVIATION_HIGH) | price_ratio.lt(PRICE_DEVIATION_LOW) | freight_ratio.gt(FREIGHT_DEVIATION_HIGH)
    return flagged.reindex(lines.index).fillna(True).astype(bool)


def excluded_purchase_rows(purchases: pd.DataFrame) -> pd.DataFrame:
    """Rows dropped from spend metrics, each with the reason, so they can be inspected."""
    data = purchases.copy()
    reasons = pd.Series("", index=data.index)
    for column in ("quantity_kg", "price_per_kg_inr", "freight_cost_to_india_per_kg_inr"):
        data[column] = pd.to_numeric(data[column], errors="coerce") if column in data else pd.NA
    checks = [
        (data["quantity_kg"].isna() | data["price_per_kg_inr"].isna() | data["freight_cost_to_india_per_kg_inr"].isna(), "non-numeric or blank quantity/price/freight"),
        (data["quantity_kg"].le(0).fillna(False), "zero or negative quantity"),
        (data["price_per_kg_inr"].lt(0).fillna(False), "negative price"),
        (data["freight_cost_to_india_per_kg_inr"].lt(0).fillna(False), "negative freight"),
    ]
    for mask, reason in checks:
        reasons = reasons.where(~mask | reasons.ne(""), reason)
    return purchases.assign(exclusion_reason=reasons)[reasons.ne("")]


def profile_data(ctx: AnalystContext) -> ToolResult:
    missing = sorted(PURCHASE_REQUIRED - set(ctx.purchases.columns))
    if missing:
        return ToolResult("profile_data", "error", {}, f"Purchase file is missing required columns: {', '.join(missing)}.")
    lines = prepare_purchase_lines(ctx.purchases)
    dates = pd.to_datetime(ctx.purchases["purchase_date"], errors="coerce").dropna()
    return ToolResult("profile_data", "ok", {}, "Data profile calculated from the loaded files.", {
        "purchase_rows": int(len(ctx.purchases)), "vendor_offer_rows": int(len(ctx.offers)),
        "valid_purchase_lines": int(len(lines)), "excluded_purchase_rows": int(len(ctx.purchases) - len(lines)),
        "quarters": ctx.quarters, "first_purchase_date": str(dates.min().date()) if not dates.empty else None,
        "last_purchase_date": str(dates.max().date()) if not dates.empty else None,
        "material_count": len(ctx.materials), "data_fingerprint": ctx.fingerprint,
    })


def check_data_quality(ctx: AnalystContext, quarter: str | None = None) -> ToolResult:
    quarter = quarter or ctx.selected_quarter
    frame = _quarter_frame(ctx, quarter)
    if frame.empty:
        return ToolResult("check_data_quality", "empty", {"quarter": quarter}, f"No purchase rows for {quarter}.")
    quality = run_quality_checks(frame, ctx.offers)
    summary = analyze_spend(frame, ctx.offers)["summary"]
    quality_flags = {key: quality[key] for key in (
        "missing_purchase_cells", "duplicate_purchase_order_ids", "duplicate_vendor_offers", "invalid_purchase_rows",
        "invalid_purchase_dates", "unmatched_purchase_rows", "offer_deviation_rows", "extreme_price_rows")}
    return ToolResult("check_data_quality", "ok", {"quarter": quarter}, "Data-quality checks calculated.", {
        "quarter": quarter, "rows_checked": quality["purchase_rows"], "flags": quality_flags,
        "findings": quality["findings"], "comparable_offer_coverage_pct": round(summary["comparable_offer_coverage_pct"], 2),
    })


def compare_spend(ctx: AnalystContext, quarter: str | None = None) -> ToolResult:
    quarter = quarter or ctx.selected_quarter
    if quarter not in ctx.quarters:
        return ToolResult("compare_spend", "error", {"quarter": quarter}, "No quarter is available to compare.")
    index = ctx.quarters.index(quarter)
    if index == 0:
        return ToolResult("compare_spend", "empty", {"quarter": quarter}, f"{quarter} is the first quarter; there is no prior quarter.")
    prior = ctx.quarters[index - 1]
    rows = {}
    for label in (quarter, prior):
        summary = analyze_spend(_quarter_frame(ctx, label), ctx.offers)["summary"]
        rows[label] = {key: summary[key] for key in (
            "landed_spend_inr", "material_spend_inr", "freight_spend_inr", "total_quantity_kg",
            "average_landed_cost_per_kg_inr", "valid_purchase_lines")}
    change = rows[quarter]["landed_spend_inr"] - rows[prior]["landed_spend_inr"]
    pct = change / rows[prior]["landed_spend_inr"] * 100 if rows[prior]["landed_spend_inr"] else None
    return ToolResult("compare_spend", "ok", {"quarter": quarter}, "Quarter-over-quarter spend compared.", {
        "quarter": quarter, "prior_quarter": prior, "current": rows[quarter], "prior": rows[prior],
        "landed_spend_change_inr": change, "landed_spend_change_pct": pct,
    })


def explain_variance(ctx: AnalystContext, quarter: str | None = None, top_n: int = 5) -> ToolResult:
    """Exact volume / price-rate / freight-rate / new-material split; the parts sum to the total change."""
    quarter = quarter or ctx.selected_quarter
    if quarter not in ctx.quarters or ctx.quarters.index(quarter) == 0:
        return ToolResult("explain_variance", "empty", {"quarter": quarter, "top_n": top_n}, "A prior quarter is required to explain a change.")
    prior = ctx.quarters[ctx.quarters.index(quarter) - 1]

    def rollup(label: str) -> pd.DataFrame:
        lines = prepare_purchase_lines(_quarter_frame(ctx, label))
        grouped = lines.groupby("material").agg(q=("quantity_kg", "sum"), m=("material_spend_inr", "sum"), f=("freight_spend_inr", "sum"))
        grouped["p"] = grouped["m"] / grouped["q"]
        grouped["fr"] = grouped["f"] / grouped["q"]
        return grouped

    cur, prev = rollup(quarter), rollup(prior)
    both = cur.index.intersection(prev.index)
    effects = pd.DataFrame(index=cur.index.union(prev.index), data={"volume": 0.0, "price": 0.0, "freight": 0.0, "new_or_dropped": 0.0})
    landed_prev_rate = prev.loc[both, "p"] + prev.loc[both, "fr"]
    effects.loc[both, "volume"] = (cur.loc[both, "q"] - prev.loc[both, "q"]) * landed_prev_rate
    effects.loc[both, "price"] = (cur.loc[both, "p"] - prev.loc[both, "p"]) * cur.loc[both, "q"]
    effects.loc[both, "freight"] = (cur.loc[both, "fr"] - prev.loc[both, "fr"]) * cur.loc[both, "q"]
    new = cur.index.difference(prev.index)
    dropped = prev.index.difference(cur.index)
    effects.loc[new, "new_or_dropped"] = cur.loc[new, "m"] + cur.loc[new, "f"]
    effects.loc[dropped, "new_or_dropped"] = -(prev.loc[dropped, "m"] + prev.loc[dropped, "f"])
    effects["net"] = effects.sum(axis=1)
    total_change = float((cur["m"] + cur["f"]).sum() - (prev["m"] + prev["f"]).sum())
    components = {name: float(effects[name].sum()) for name in ("volume", "price", "freight", "new_or_dropped")}
    reconciliation_delta = total_change - sum(components.values())
    top = effects.reindex(effects["net"].abs().sort_values(ascending=False).index).head(top_n)
    return ToolResult("explain_variance", "ok", {"quarter": quarter, "top_n": top_n}, "Spend change decomposed.", {
        "quarter": quarter, "prior_quarter": prior, "landed_spend_change_inr": total_change,
        "components_inr": components, "reconciliation_delta_inr": reconciliation_delta,
        "reconciles": abs(reconciliation_delta) < 0.01,
        "top_materials": [{"material": material, **{k: float(row[k]) for k in ("volume", "price", "freight", "new_or_dropped", "net")}}
                          for material, row in top.iterrows()],
    })


def rank_savings_opportunities(ctx: AnalystContext, quarter: str | None = None, group_by: str = "material", top_n: int = 5) -> ToolResult:
    quarter = quarter or ctx.selected_quarter
    frame = _quarter_frame(ctx, quarter)
    analysis = analyze_spend(frame, ctx.offers)
    lines = analysis["savings_lines"]
    args = {"quarter": quarter, "group_by": group_by, "top_n": top_n}
    if lines.empty:
        return ToolResult("rank_savings_opportunities", "empty", args, f"No valid purchase lines for {quarter}.")
    column = GROUP_COLUMNS[group_by]
    lines = lines.assign(unreliable=_flag_lines(lines, ctx.offers))
    grouped = lines.groupby(column).agg(
        combined_saving_inr=("combined_saving_inr", "sum"), price_saving_inr=("price_saving_inr", "sum"),
        freight_saving_inr=("freight_saving_inr", "sum"), landed_spend_inr=("landed_spend_inr", "sum"),
        purchase_lines=("quantity_kg", "size"), unreliable_lines=("unreliable", "sum"),
    ).sort_values("combined_saving_inr", ascending=False)
    total = float(lines["combined_saving_inr"].sum())
    summary = analysis["summary"]
    return ToolResult("rank_savings_opportunities", "ok", args, "Savings opportunities ranked.", {
        "quarter": quarter, "group_by": group_by, "total_combined_saving_inr": total,
        "total_landed_spend_inr": summary["landed_spend_inr"], "comparable_offer_coverage_pct": round(summary["comparable_offer_coverage_pct"], 2),
        "groups_total": int(len(grouped)),
        "top": [{"name": str(name), "combined_saving_inr": float(row["combined_saving_inr"]),
                 "price_saving_inr": float(row["price_saving_inr"]), "freight_saving_inr": float(row["freight_saving_inr"]),
                 "landed_spend_inr": float(row["landed_spend_inr"]), "purchase_lines": int(row["purchase_lines"]),
                 "unreliable_lines": int(row["unreliable_lines"]),
                 "share_of_total_saving_pct": float(row["combined_saving_inr"] / total * 100) if total else 0.0}
                for name, row in grouped.head(top_n).iterrows()],
    })


def lookup_best_quote(ctx: AnalystContext, material: str, country: str | None = None) -> ToolResult:
    args = {"material": material, "country": country}
    offers = ctx.offers.copy()
    for column in ("price_per_kg_inr", "freight_cost_to_india_per_kg_inr"):
        offers[column] = pd.to_numeric(offers[column], errors="coerce")
    offers = offers.dropna(subset=["price_per_kg_inr", "freight_cost_to_india_per_kg_inr"])
    offers = offers[(offers["price_per_kg_inr"] >= 0) & (offers["freight_cost_to_india_per_kg_inr"] >= 0)]
    offers = offers[offers["material"].astype(str).str.casefold().eq(material.casefold())]
    if country:
        offers = offers[offers["country"].astype(str).str.casefold().eq(country.casefold())]
    if offers.empty:
        return ToolResult("lookup_best_quote", "empty", args, f"No valid quotes for {material}{' from ' + country if country else ''}.")
    offers = offers.assign(landed=offers["price_per_kg_inr"] + offers["freight_cost_to_india_per_kg_inr"]).sort_values(["landed", "vendor"])
    return ToolResult("lookup_best_quote", "ok", args, "Lowest landed-cost quotes found.", {
        "material": str(offers.iloc[0]["material"]), "country_filter": country, "quotes_considered": int(len(offers)),
        "best": [{"vendor": str(r["vendor"]), "country": str(r["country"]), "price_per_kg_inr": float(r["price_per_kg_inr"]),
                  "freight_per_kg_inr": float(r["freight_cost_to_india_per_kg_inr"]), "landed_per_kg_inr": float(r["landed"])}
                 for _, r in offers.head(3).iterrows()],
    })


# name -> (callable, description shown to the planner, allowed params)
TOOLS: dict[str, tuple[Callable[..., ToolResult], str, dict[str, str]]] = {
    "profile_data": (profile_data, "Row counts, quarters, date range, schema status.", {}),
    "check_data_quality": (check_data_quality, "Data-quality flags and quote coverage for a quarter.", {"quarter": "quarter"}),
    "compare_spend": (compare_spend, "Landed spend vs the prior quarter.", {"quarter": "quarter"}),
    "explain_variance": (explain_variance, "Split a spend change into volume, price, freight and new-material effects.", {"quarter": "quarter", "top_n": "int1-10"}),
    "rank_savings_opportunities": (rank_savings_opportunities, "Rank savings by material, vendor or category with data-reliability counts.",
                                   {"quarter": "quarter", "group_by": "enum:material|vendor|category", "top_n": "int1-10"}),
    "lookup_best_quote": (lookup_best_quote, "Lowest landed-cost quotes for one material, optional country.", {"material": "material!", "country": "country"}),
}


def validate_call(ctx: AnalystContext, name: Any, args: Any) -> dict[str, Any]:
    """Return cleaned arguments or raise ToolArgError; unknown tools and parameters are rejected."""
    if name not in TOOLS:
        raise ToolArgError(f"tool '{name}' is not on the allowlist")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolArgError("arguments must be an object")
    spec = TOOLS[name][2]
    unknown = set(args) - set(spec)
    if unknown:
        raise ToolArgError(f"unknown parameter(s) for {name}: {', '.join(sorted(map(str, unknown)))}")
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if value is None:
            continue
        kind = spec[key]
        if kind == "quarter":
            if str(value) not in ctx.quarters:
                raise ToolArgError(f"quarter '{value}' is not in the data")
            clean[key] = str(value)
        elif kind.startswith("int"):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
                raise ToolArgError(f"{key} must be an integer from 1 to 10")
            clean[key] = value
        elif kind.rstrip("!") == "material":
            match = next((m for m in ctx.materials if m.casefold() == str(value).casefold()), None)
            if match is None:
                raise ToolArgError(f"material '{value}' is not in the data")
            clean[key] = match
        elif kind == "country":
            match = next((c for c in ctx.countries if c.casefold() == str(value).casefold()), None)
            if match is None:
                raise ToolArgError(f"country '{value}' is not in the offers")
            clean[key] = match
        else:
            allowed = kind.removeprefix("enum:").split("|")
            if value not in allowed:
                raise ToolArgError(f"{key} must be one of {', '.join(allowed)}")
            clean[key] = value
    for key, kind in spec.items():
        if kind.endswith("!") and key not in clean:
            raise ToolArgError(f"{name} requires '{key}'")
    return clean


def run_tool(ctx: AnalystContext, name: str, args: dict[str, Any]) -> ToolResult:
    """Validate then execute one tool; failures return an error result instead of raising."""
    try:
        clean = validate_call(ctx, name, args)
        return TOOLS[name][0](ctx, **clean)
    except ToolArgError as exc:
        return ToolResult(str(name), "error", dict(args) if isinstance(args, dict) else {}, f"Invalid request: {exc}.")
    except Exception as exc:  # a tool bug must not crash the dashboard
        return ToolResult(str(name), "error", {}, f"Tool failed: {type(exc).__name__}.")
