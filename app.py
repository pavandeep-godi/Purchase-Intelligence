"""Streamlit purchase intelligence dashboard for chemicals procurement."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from purchase_intelligence.agents.data_quality_agent import run_quality_checks
from purchase_intelligence.agents.groq_analysis_agent import build_groq_context, load_groq_environment, run_groq_review
from purchase_intelligence.agents.spend_analysis_agent import analyze_spend

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="purchase_intelligence", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#162b3b; --muted:#657987; --teal:#087e79; --teal-light:#e5f4f1; --border:#e4ecef; }
html, body, [class*="css"] { font-family:system-ui,sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(180deg,#f3f7f8 0%,#f8fafb 260px,#f6f9fa 100%); }
.block-container { padding-top:5rem; padding-bottom:2.5rem; max-width:1440px; }
h1,h2,h3 { font-family:system-ui,sans-serif; color:var(--ink); letter-spacing:-.025em; }
.hero { background:linear-gradient(115deg,#123247 0%,#0d5c62 100%); border-radius:20px; padding:28px 34px; color:white; margin:0 0 18px; box-shadow:0 14px 34px rgba(19,53,69,.14); position:relative; overflow:hidden; }
.hero h1 { color:white; margin:0; font-size:2rem; letter-spacing:-.04em; }
.hero p { color:#d5e9e8; margin:7px 0 0; font-size:.98rem; }
.eyebrow { color:#91dbcc; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; margin-bottom:7px; }
.metric-card { background:linear-gradient(145deg,#fff 0%,#fbfdfd 100%); border:1px solid var(--border); border-radius:15px; padding:15px 17px; height:146px; display:flex; flex-direction:column; box-shadow:0 4px 14px rgba(22,43,59,.045); overflow:hidden; transition:transform .16s ease,box-shadow .16s ease; }
.metric-card:hover { transform:translateY(-2px); box-shadow:0 9px 20px rgba(22,43,59,.08); }
.metric-label { color:var(--muted); font-size:.77rem; font-weight:650; line-height:1.25; min-height:2.5em; margin-bottom:5px; }
.metric-value { color:var(--ink); font-family:system-ui,sans-serif; font-weight:800; font-size:clamp(1.12rem,1.6vw,1.5rem); line-height:1.15; min-height:1.3em; white-space:nowrap; }
.metric-help { color:#81929c; font-size:.7rem; line-height:1.25; margin-top:auto; min-height:2.5em; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.section-note { color:var(--muted); font-size:.9rem; margin-top:-8px; margin-bottom:15px; }
.callout { border-left:4px solid var(--teal); background:var(--teal-light); padding:14px 16px; border-radius:0 10px 10px 0; color:#20434a; margin:8px 0 16px; }
div[data-testid="stTabs"] [role="tablist"] { display:flex; gap:16px; border-bottom:1px solid var(--border); padding-bottom:6px; }
div[data-testid="stTabs"] [role="tab"] { flex:1 1 0% !important; width:auto !important; min-width:0; justify-content:center; font-weight:700; border-radius:10px; padding:.8rem 1rem; min-height:48px; }
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--teal); background:#e8f4f2; }
div[data-testid="stTabs"] [role="tabpanel"] { padding-top:1.1rem; }
.kpi-section-gap { height:22px; }
[data-testid="stExpander"] { border-color:var(--border); border-radius:12px; background:rgba(255,255,255,.7); }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid var(--border); border-radius:14px; padding:8px 8px 2px; box-shadow:0 3px 12px rgba(22,43,59,.035); }
[data-testid="stMetric"] { background:white; border:1px solid var(--border); border-radius:12px; padding:13px 16px; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; overflow:hidden; }
footer { visibility:hidden; }
@media (max-width: 1100px) {
    .block-container { max-width:100%; padding-left:2rem; padding-right:2rem; }
    .hero { padding:24px 28px; }
    div[data-testid="stTabs"] [role="tablist"] { gap:8px; }
    div[data-testid="stTabs"] [role="tab"] { padding:.7rem .6rem; font-size:.9rem; }
    [data-testid="stPlotlyChart"] { padding:5px 4px 1px; }
}
@media (max-width: 700px) {
    .block-container { padding:4.8rem .8rem 1.5rem; }
    .hero { border-radius:15px; padding:20px 20px; margin-bottom:14px; }
    .hero h1 { font-size:1.55rem; }
    .hero p { font-size:.88rem; line-height:1.4; }
    .eyebrow { font-size:.62rem; letter-spacing:.1em; }
    div[data-testid="stTabs"] [role="tablist"] { flex-wrap:wrap; gap:5px; }
    div[data-testid="stTabs"] [role="tab"] { flex:1 1 calc(33.333% - 5px) !important; min-width:0; min-height:52px; padding:.55rem .25rem; font-size:.72rem; line-height:1.2; white-space:normal; }
    div[data-testid="stTabs"] [role="tabpanel"] { padding-top:.75rem; }
    .metric-card { height:auto; min-height:122px; padding:13px 13px; border-radius:12px; }
    .metric-label { font-size:.72rem; min-height:2.2em; }
    .metric-value { font-size:clamp(1rem,5vw,1.3rem); white-space:normal; overflow-wrap:anywhere; }
    .metric-help { font-size:.66rem; min-height:2.4em; }
    .section-note { font-size:.82rem; margin-top:-5px; margin-bottom:11px; }
    .callout { padding:11px 12px; font-size:.88rem; }
    .kpi-section-gap { height:14px; }
    [data-testid="stPlotlyChart"] { border-radius:11px; margin-bottom:.35rem; }
    .js-plotly-plot .bartext { display:none !important; }
    [data-testid="stDataFrame"] { max-width:100%; overflow-x:auto; }
    [data-testid="stExpander"] summary { font-size:.9rem; }
}
</style>
""", unsafe_allow_html=True)


def money(value: float, compact: bool = False) -> str:
    if compact and abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:,.1f} Cr"
    if compact and abs(value) >= 100_000:
        return f"₹{value / 100_000:,.1f} L"
    return f"₹{value:,.0f}"


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    purchase_upload = st.session_state.get("purchase_upload")
    offer_upload = st.session_state.get("offer_upload")
    purchases = pd.read_csv(purchase_upload) if purchase_upload else pd.read_csv(DATA_DIR / "purchase_history.csv")
    offers = pd.read_csv(offer_upload) if offer_upload else pd.read_csv(DATA_DIR / "vendor_offers.csv")
    if purchase_upload and offer_upload:
        source = "Uploaded data"
    elif purchase_upload:
        source = "Uploaded purchases + sample offers"
    elif offer_upload:
        source = "Sample purchases + uploaded offers"
    else:
        source = "Included mock data"
    return purchases, offers, source


def currency_axis(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title, title_font_size=16, title_font_color="#193749",
        plot_bgcolor="white", paper_bgcolor="white", font_color="#526874", margin=dict(l=14, r=14, t=56, b=14),
        legend_title_text="", hoverlabel=dict(bgcolor="white"),
    )
    fig.update_yaxes(gridcolor="#edf1f3", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def money_label(value: float) -> str:
    """Format chart labels compactly while retaining familiar Indian currency units."""
    absolute = abs(value)
    if absolute >= 10_000_000:
        return f"₹{value / 10_000_000:.1f} Cr"
    if absolute >= 100_000:
        return f"₹{value / 100_000:.1f} L"
    if absolute >= 1_000:
        return f"₹{value / 1_000:.0f}K"
    return f"₹{value:,.0f}"


def show_chart(fig: go.Figure) -> None:
    """Render an interactive chart without Plotly's extra toolbar buttons."""
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_groq_review(context_json: str, credential_fingerprint: str, model_name: str, review_logic_version: str) -> dict[str, object]:
    """Avoid repeating the full Groq analysis during routine Streamlit reruns."""
    return run_groq_review(json.loads(context_json))


st.markdown(
    '<div class="hero"><div class="eyebrow">CHEMICALS PROCUREMENT · SPEND INTELLIGENCE</div>'
    '<h1>purchase_intelligence</h1>'
    '<p>Executive spend, supplier, category, and savings signals for business leaders.</p></div>',
    unsafe_allow_html=True,
)
purchases, offers, data_source = load_source_data()
quality = run_quality_checks(purchases, offers)
load_groq_environment()

quarter_choices = sorted(purchases["quarter"].dropna().astype(str).unique()) if "quarter" in purchases else []
selected_quarter = quarter_choices[-1] if quarter_choices else "All quarters"

if quarter_choices:
    filter_columns = st.columns([1.1, 1.6, 3.3])
    with filter_columns[0]:
        selected_quarter = st.selectbox(
            "Reporting quarter", quarter_choices, index=len(quarter_choices) - 1,
            help="Spend bars compare this quarter with the two preceding quarters; the line chart keeps full history and highlights this quarter.",
        )
    with filter_columns[1]:
        selected_index = quarter_choices.index(selected_quarter)
        preview_quarters = quarter_choices[max(0, selected_index - 2):selected_index + 1]
        st.caption(f"Spend bars: {preview_quarters[0]}–{preview_quarters[-1]}")
    with filter_columns[2]:
        st.caption(f"{data_source} · {len(purchases):,} purchase lines")
else:
    st.caption(f"{data_source} · {len(purchases):,} purchase lines · {len(offers):,} vendor offers")

if "quarter" in purchases and quarter_choices:
    selected_index = quarter_choices.index(selected_quarter)
    comparison_quarters = quarter_choices[max(0, selected_index - 2):selected_index + 1]
    filtered = purchases[purchases["quarter"].astype(str).eq(selected_quarter)].copy()
    selected_quarters = [selected_quarter]
else:
    comparison_quarters = quarter_choices
    selected_quarters = quarter_choices
    filtered = purchases.copy()
analysis = analyze_spend(filtered, offers)
comparison_population = purchases[purchases["quarter"].astype(str).isin(comparison_quarters)].copy() if "quarter" in purchases else purchases
comparison_analysis = analyze_spend(comparison_population, offers)
history_analysis = analyze_spend(purchases, offers)
quarter_quality = run_quality_checks(filtered, offers)
groq_context = build_groq_context(analysis, quarter_quality, offers)
credential_fingerprint = hashlib.sha256(os.getenv("GROQ_API_KEY", "").encode("utf-8")).hexdigest()
model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
llm_review = cached_groq_review(
    json.dumps(groq_context, ensure_ascii=False, sort_keys=True), credential_fingerprint, model_name, "groq-review-v2"
)
summary = dict(analysis["summary"])
summary.update(llm_review["dashboard_metrics"])
lines = analysis["lines"]
savings = analysis["savings_lines"]
dashboard_series = llm_review["dashboard_series"]
vendors = analysis["vendors"].copy()
llm_vendor_metrics = pd.DataFrame(dashboard_series["vendors"])
if not vendors.empty and not llm_vendor_metrics.empty:
    llm_vendor_metrics = llm_vendor_metrics.set_index("vendor")
    for metric in ["landed_spend_inr", "quantity_kg", "combined_savings_inr", "average_landed_cost_per_kg_inr", "spend_share_pct"]:
        vendors[metric] = vendors["vendor"].map(llm_vendor_metrics[metric])
quarterly = comparison_analysis["quarterly"]
quarterly_history = history_analysis["quarterly"]
categories = analysis["categories"]
materials = analysis["savings_lines"].groupby("material", as_index=False).agg(
    price_savings_inr=("price_saving_inr", "sum"), freight_savings_inr=("freight_saving_inr", "sum"),
    combined_savings_inr=("combined_saving_inr", "sum"),
) if not analysis["savings_lines"].empty else pd.DataFrame()
narrative = llm_review["executive_summary"]
st.markdown(f"**Reporting period:** {', '.join(selected_quarters) if selected_quarters else 'No quarters selected'} &nbsp; · &nbsp; **Data:** {data_source}")
if summary["excluded_purchase_rows"]:
    st.warning(f"{summary['excluded_purchase_rows']:,} invalid purchase rows were excluded from spend calculations. Review Data checks below.")
if llm_review["source"].startswith("Python fallback"):
    if "GROQ_API_KEY not found" in llm_review["status"]:
        st.warning(
            "Groq review is unavailable because no API key was found. Dashboard metrics are using the validated Python fallback. "
            "For Streamlit Community Cloud, open this app's Settings → Secrets and add GROQ_API_KEY = \"your-key-here\", then reboot the app."
        )
    else:
        st.warning(f"Groq review is unavailable: {llm_review['status']}. Dashboard metrics are using the validated Python fallback.")
else:
    st.caption(f"Groq analysis active · {llm_review['matched_metric_count']}/{llm_review['metric_count']} KPI values reconciled · "
               f"Charts: {llm_review['dashboard_series_source']}.")

spend_tab, vendor_tab, savings_tab = st.tabs(["Spend overview", "Vendor summary", "Savings opportunities"])

with spend_tab:
    st.subheader("Spend at a glance")
    st.markdown(f'<div class="section-note">Selected quarter: {selected_quarter}. Spend bars compare the selected quarter with up to two prior quarters; the line chart shows full history and highlights the selection.</div>', unsafe_allow_html=True)
    quarter_rows = quarterly.set_index("quarter") if not quarterly.empty else pd.DataFrame()
    current_quarter_row = quarter_rows.loc[selected_quarter] if not quarter_rows.empty and selected_quarter in quarter_rows.index else None
    previous_quarter = quarter_choices[quarter_choices.index(selected_quarter) - 1] if quarter_choices and quarter_choices.index(selected_quarter) > 0 else None
    previous_quarter_row = quarter_rows.loc[previous_quarter] if previous_quarter and not quarter_rows.empty and previous_quarter in quarter_rows.index else None
    selected_spend = float(current_quarter_row["landed_spend_inr"]) if current_quarter_row is not None else float(summary["landed_spend_inr"])
    previous_spend = float(previous_quarter_row["landed_spend_inr"]) if previous_quarter_row is not None else None
    spend_change_pct = ((selected_spend / previous_spend) - 1) * 100 if previous_spend else None
    metric_cols = st.columns(4)
    metric_items = [
        ("Landed spend", money(selected_spend, True), selected_quarter),
        ("Spend vs. prior quarter", f"{spend_change_pct:+.1f}%" if spend_change_pct is not None else "—", f"Prior quarter: {money(previous_spend, True)}" if previous_spend is not None else "No prior quarter"),
        ("Average landed cost", f"₹{summary['average_landed_cost_per_kg_inr']:,.2f}/kg", "Weighted by purchased quantity"),
        ("Freight share", f"{summary['freight_share_pct']:.1f}%", "Of selected-quarter landed spend"),
    ]
    for column, (label, value, hint) in zip(metric_cols, metric_items):
        with column:
            metric_card(label, value, hint)
    st.markdown('<div class="kpi-section-gap"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="callout"><b>Groq procurement summary:</b> {narrative}</div>', unsafe_allow_html=True)
    if llm_review["dashboard_insights"]:
        with st.expander("Groq review: notable spend insights"):
            for insight in llm_review["dashboard_insights"]:
                st.write(f"• {insight}")

    if not quarterly.empty:
        spend_chart = px.bar(quarterly, x="quarter", y=["material_spend_inr", "freight_spend_inr"],
                             labels={"quarter": "Quarter", "value": "Spend (INR)", "variable": "Spend type",
                                     "material_spend_inr": "Material", "freight_spend_inr": "Freight"},
                             color_discrete_map={"material_spend_inr": "#147b78", "freight_spend_inr": "#91c9bd"},
                             barmode="stack")
        spend_chart.update_traces(hovertemplate="%{x}<br>%{fullData.name}: ₹%{y:,.0f}<extra></extra>")
        spend_chart.add_trace(go.Scatter(
            x=quarterly["quarter"], y=quarterly["landed_spend_inr"], mode="text",
            text=[money_label(value) for value in quarterly["landed_spend_inr"]],
            textposition="top center", textfont=dict(color="#193749", size=11),
            showlegend=False, hoverinfo="skip", cliponaxis=False,
        ))
        spend_chart = currency_axis(spend_chart, "Spend comparison · selected + prior quarters")
        max_landed_spend = float(quarterly["landed_spend_inr"].max()) if not quarterly.empty else 0.0
        spend_chart.update_yaxes(tickprefix="₹", separatethousands=True, range=[0, max_landed_spend * 1.18] if max_landed_spend else None)
        spend_chart.for_each_trace(lambda trace: trace.update(name={"material_spend_inr": "Material", "freight_spend_inr": "Freight"}.get(trace.name, trace.name)))
        spend_chart.update_layout(yaxis_title="Spend (INR)", xaxis_title="Quarter")
        show_chart(spend_chart)

    if not quarterly_history.empty:
        history = quarterly_history.copy()
        history["quarter_label"] = history["quarter"].astype(str).map(lambda quarter: f"Q{quarter[-1]} '{quarter[2:4]}")
        selected_history = history[history["quarter"].astype(str).eq(selected_quarter)]
        cost_chart = go.Figure()
        cost_chart.add_trace(go.Scatter(
            x=history["quarter_label"], y=history["average_landed_cost_per_kg_inr"],
            mode="lines", name="Historical trend",
            line=dict(color="#168b80", width=3),
            hovertemplate="%{x}<br>₹%{y:,.2f}/kg<extra></extra>",
        ))
        if not selected_history.empty:
            cost_chart.add_trace(go.Scatter(
                x=selected_history["quarter_label"], y=selected_history["average_landed_cost_per_kg_inr"],
                mode="markers", name="Selected quarter",
                marker=dict(color="#e17c42", size=14, line=dict(color="white", width=2)),
                hovertemplate="Selected quarter · %{x}<br>₹%{y:,.2f}/kg<extra></extra>",
            ))
        cost_chart = currency_axis(cost_chart, "Average landed cost per kg · all quarters")
        cost_chart.update_yaxes(tickprefix="₹", separatethousands=True, title="INR per kg")
        cost_chart.update_xaxes(title="Quarter", tickangle=0, categoryorder="array", categoryarray=history["quarter_label"].tolist())
        cost_chart.update_layout(showlegend=False, margin=dict(l=14, r=14, t=48, b=12), height=340)
        show_chart(cost_chart)
        st.caption(f"Unfiltered quarterly history. The selected quarter, {selected_quarter}, is marked in orange.")
    st.markdown("#### Category and material view")
    st.markdown('<div class="section-note">Selected-quarter category mix and largest material commitments.</div>', unsafe_allow_html=True)
    category_analysis = analysis
    categories = category_analysis["categories"]
    if not categories.empty and summary["landed_spend_inr"]:
        categories["spend_share_pct"] = categories["landed_spend_inr"] / summary["landed_spend_inr"] * 100
    materials = category_analysis["savings_lines"].groupby("material", as_index=False).agg(
        price_savings_inr=("price_saving_inr", "sum"), freight_savings_inr=("freight_saving_inr", "sum"),
        combined_savings_inr=("combined_saving_inr", "sum"),
    ) if not category_analysis["savings_lines"].empty else pd.DataFrame()
    if not categories.empty:
        top_category = categories.sort_values("landed_spend_inr", ascending=False).iloc[0]
        top_material_spend = category_analysis["lines"].groupby("material")["landed_spend_inr"].sum().sort_values(ascending=False)
        leader_metrics = st.columns(3)
        leader_items = [
            ("Leading category spend", money(float(top_category["landed_spend_inr"]), True), str(top_category["chemical_category"])),
            ("Leading category share", f"{top_category['spend_share_pct']:.1f}%", "Of company landed spend"),
            ("Leading material", str(top_material_spend.index[0]) if not top_material_spend.empty else "—",
             money(float(top_material_spend.iloc[0]), True) if not top_material_spend.empty else "No selected-category spend"),
        ]
        for column, (label, value, hint) in zip(leader_metrics, leader_items):
            with column:
                metric_card(label, value, hint)
        st.markdown('<div class="kpi-section-gap"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.1, 1])
        with left:
            category_chart = px.bar(categories.sort_values("landed_spend_inr"), x="landed_spend_inr", y="chemical_category",
                                    orientation="h", color="landed_spend_inr", color_continuous_scale=["#cce8e2", "#087e79"],
                                    labels={"landed_spend_inr": "Landed spend (INR)", "chemical_category": "Chemical category"})
            category_chart.update_layout(coloraxis_showscale=False)
            category_chart.update_traces(
                text=[money_label(value) for value in categories.sort_values("landed_spend_inr")["landed_spend_inr"]],
                textposition="outside", textfont=dict(size=10, color="#193749"), cliponaxis=False,
                hovertemplate="%{y}<br>Landed spend: ₹%{x:,.0f}<extra></extra>",
            )
            category_chart = currency_axis(category_chart, "Which categories account for spend?")
            category_chart.update_xaxes(tickprefix="₹", separatethousands=True)
            category_chart.update_xaxes(range=[0, float(categories["landed_spend_inr"].max()) * 1.2])
            category_chart.update_yaxes(tickprefix="")
            category_chart.update_layout(xaxis_title="Landed spend (INR)", yaxis_title="")
            show_chart(category_chart)
        with right:
            st.markdown("#### Analysis checks")
            checks = analysis["validations"]
            for label, key in [("Landed spend reconciles", "landed_spend_equals_material_plus_freight"),
                               ("Savings recompute matches", "combined_savings_recomputed_from_unit_rates"),
                               ("Purchase row count is valid", "valid_lines_do_not_exceed_file_rows")]:
                st.write(f"{'✅' if checks[key] else '❌'} {label}")
            st.caption("Groq reviews and produces dashboard metric values from the full-data rollups; Python independently reconciles the displayed figures.")
    with st.expander("Data checks and potential anomalies"):
        st.caption(f"Checked {quality['purchase_rows']:,} purchase rows and {quality['vendor_offer_rows']:,} vendor offers in the uploaded/source data.")
        for finding in quality["findings"]:
            st.write(f"• {finding}")
        if llm_review["additional_validation_findings"]:
            st.markdown("**Groq's additional checks — review before action**")
            for finding in llm_review["additional_validation_findings"]:
                st.write(f"• {finding}")
        with st.expander("Groq metric reconciliation"):
            st.caption("LLM-generated KPI and chart values are shown only when every value reconciles to Python's row-level calculation. Mismatches use the Python result.")
            for key, audit in llm_review["metric_audit"].items():
                st.write(f"{'✅' if audit['matched'] else '↩️'} {key}: Groq={audit['llm_value']} · Python={audit['python_value']}")
            for check in llm_review.get("calculation_checks", []):
                if isinstance(check, dict):
                    st.write(f"• {check.get('name', 'Check')}: {check.get('status', 'review')} — {check.get('explanation', '')}")
        if summary["excluded_purchase_rows"]:
            st.write(f"Excluded from KPIs: {summary['excluded_purchase_rows']:,} invalid purchase lines.")

with vendor_tab:
    st.subheader("Vendor performance")
    st.markdown(f'<div class="section-note">Supplier performance and concentration for {selected_quarter}; opportunity estimates use same-material and source-country quotes.</div>', unsafe_allow_html=True)
    if vendors.empty:
        st.info("No valid purchase lines are available with the selected filters.")
    else:
        top_vendor = vendors.sort_values("landed_spend_inr", ascending=False).iloc[0]
        top_three_share = float(vendors.nlargest(3, "landed_spend_inr")["spend_share_pct"].sum())
        vendor_savings = float(vendors["combined_savings_inr"].sum())
        vendor_cols = st.columns(4)
        vendor_kpis = [
            ("Active suppliers", f"{summary['vendor_count']}", selected_quarter),
            ("Top supplier share", f"{top_vendor['spend_share_pct']:.1f}%", str(top_vendor["vendor"])),
            ("Top 3 supplier share", f"{top_three_share:.1f}%", "Spend concentration indicator"),
            ("Supplier opportunity", money(vendor_savings, True), "Indicative landed-cost savings"),
        ]
        for column, (label, value, hint) in zip(vendor_cols, vendor_kpis):
            with column:
                metric_card(label, value, hint)
        st.markdown('<div class="kpi-section-gap"></div>', unsafe_allow_html=True)
        spend_by_vendor, savings_by_vendor = st.columns(2)
        with spend_by_vendor:
            show = vendors.nlargest(12, "landed_spend_inr").sort_values("landed_spend_inr")
            fig = px.bar(show, x="landed_spend_inr", y="vendor", orientation="h",
                         color_discrete_sequence=["#147b78"],
                         labels={"landed_spend_inr": "Landed spend (INR)", "vendor": "Supplier"})
            fig.update_traces(text=[money_label(value) for value in show["landed_spend_inr"]], textposition="outside",
                              textfont=dict(size=10, color="#193749"), cliponaxis=False)
            fig.update_traces(hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>")
            fig = currency_axis(fig, "Largest suppliers by spend")
            fig.update_xaxes(tickprefix="₹", separatethousands=True)
            fig.update_xaxes(range=[0, float(show["landed_spend_inr"].max()) * 1.2])
            fig.update_yaxes(tickprefix="")
            fig.update_layout(xaxis_title="Landed spend (INR)", yaxis_title="", height=390)
            show_chart(fig)
        with savings_by_vendor:
            savings_rank = vendors.nlargest(12, "combined_savings_inr").sort_values("combined_savings_inr")
            fig = px.bar(savings_rank, x="combined_savings_inr", y="vendor", orientation="h",
                         color_discrete_sequence=["#e28b45"],
                         labels={"combined_savings_inr": "Indicative savings (INR)", "vendor": "Supplier"})
            fig.update_traces(text=[money_label(value) for value in savings_rank["combined_savings_inr"]], textposition="outside",
                              textfont=dict(size=10, color="#193749"), cliponaxis=False)
            fig.update_traces(hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>")
            fig = currency_axis(fig, "Largest supplier savings opportunities")
            fig.update_xaxes(tickprefix="₹", separatethousands=True)
            fig.update_xaxes(range=[0, float(savings_rank["combined_savings_inr"].max()) * 1.2])
            fig.update_yaxes(tickprefix="")
            fig.update_layout(xaxis_title="Indicative savings (INR)", yaxis_title="", height=390)
            show_chart(fig)
        st.markdown("#### Vendor scorecard")
        vendor_table = vendors.sort_values("landed_spend_inr", ascending=False).rename(columns={
            "vendor": "Vendor", "landed_spend_inr": "Landed spend (₹)", "spend_share_pct": "Spend share (%)",
            "average_landed_cost_per_kg_inr": "Avg landed cost (₹/kg)", "purchase_lines": "Purchase lines",
            "combined_savings_inr": "Indicative savings (₹)", "quantity_kg": "Quantity (kg)",
        })[["Vendor", "Landed spend (₹)", "Spend share (%)", "Avg landed cost (₹/kg)", "Purchase lines", "Indicative savings (₹)", "Quantity (kg)"]]
        st.dataframe(vendor_table, width="stretch", hide_index=True,
                     column_config={"Landed spend (₹)": st.column_config.NumberColumn(format="₹%.0f"),
                                    "Spend share (%)": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Avg landed cost (₹/kg)": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Indicative savings (₹)": st.column_config.NumberColumn(format="₹%.0f"),
                                    "Quantity (kg)": st.column_config.NumberColumn(format="%,.0f")})

with savings_tab:
    st.subheader("Savings opportunities")
    st.markdown('<div class="section-note">See how much could be saved by improving material price, freight, or the combined delivered cost. Benchmarks match material and source country.</div>', unsafe_allow_html=True)
    price_saving = float(savings["price_saving_inr"].sum()) if not savings.empty else 0.0
    freight_saving = float(savings["freight_saving_inr"].sum()) if not savings.empty else 0.0
    combined_saving = float(savings["combined_saving_inr"].sum()) if not savings.empty else 0.0
    potential_cols = st.columns(4)
    savings_share = combined_saving / summary["landed_spend_inr"] * 100 if summary["landed_spend_inr"] else 0.0
    potential_kpis = [
        ("Combined cost opportunity", money(combined_saving, True), "Lowest comparable landed quote"),
        ("Material price opportunity", money(price_saving, True), "Price improvement only"),
        ("Freight opportunity", money(freight_saving, True), "Freight improvement only"),
        ("Opportunity vs. spend", f"{savings_share:.1f}%", "Combined opportunity ÷ landed spend"),
    ]
    for column, (label, value, hint) in zip(potential_cols, potential_kpis):
        with column:
            metric_card(label, value, hint)
    st.markdown('<div class="kpi-section-gap"></div>', unsafe_allow_html=True)
    st.info("These are indicative quote-based opportunities, not guaranteed savings. Switching costs, quality, capacity, lead time, taxes, FX, and contract terms are not included.", icon="ℹ️")
    st.caption("Price-only and freight-only estimates are separate views and may overlap. Do not add them together. The combined opportunity compares against the lowest single quoted landed-cost offer.")

    if savings.empty or not savings["has_comparable_offer"].any():
        st.info("No same-material, same-country offer comparisons are available for this selection.")
    else:
        opportunity_by_material = materials.rename(columns={
            "price_savings_inr": "price_saving_inr", "freight_savings_inr": "freight_saving_inr",
            "combined_savings_inr": "combined_saving_inr",
        }).sort_values("combined_saving_inr", ascending=False).head(12)
        left, right = st.columns([1.2, 1])
        with left:
            chart_data = opportunity_by_material.melt(id_vars="material", value_vars=["price_saving_inr", "freight_saving_inr"],
                                                      var_name="Opportunity type", value_name="Savings (INR)")
            chart_data["Opportunity type"] = chart_data["Opportunity type"].map({"price_saving_inr": "Material price", "freight_saving_inr": "Freight"})
            fig = px.bar(chart_data, x="Savings (INR)", y="material", color="Opportunity type", barmode="group", orientation="h",
                         color_discrete_map={"Material price": "#087e79", "Freight": "#e4a35c"})
            fig.for_each_trace(lambda trace: trace.update(
                text=[money_label(value) for value in trace.x], textposition="outside",
                textfont=dict(size=9, color="#193749"), cliponaxis=False,
            ))
            fig = currency_axis(fig, "Where are the biggest price and freight opportunities?")
            fig.update_xaxes(tickprefix="₹", separatethousands=True)
            chart_max = float(chart_data["Savings (INR)"].max()) if not chart_data.empty else 0.0
            fig.update_xaxes(range=[0, chart_max * 1.25] if chart_max else None)
            fig.update_yaxes(tickprefix="")
            fig.update_layout(xaxis_title="Indicative savings (INR)", yaxis_title="Material", legend_title_text="")
            show_chart(fig)
        with right:
            top_opportunities = opportunity_by_material.sort_values("combined_saving_inr").tail(10)
            fig = px.bar(top_opportunities, x="combined_saving_inr", y="material", orientation="h",
                         color_discrete_sequence=["#168b7e"], labels={"combined_saving_inr": "Combined landed-cost opportunity (INR)", "material": "Material"})
            fig.update_traces(text=[money_label(value) for value in top_opportunities["combined_saving_inr"]], textposition="outside",
                              textfont=dict(size=10, color="#193749"), cliponaxis=False)
            fig = currency_axis(fig, "Combined opportunity by material")
            fig.update_xaxes(tickprefix="₹", separatethousands=True)
            fig.update_xaxes(range=[0, float(top_opportunities["combined_saving_inr"].max()) * 1.2])
            fig.update_yaxes(tickprefix="")
            fig.update_layout(xaxis_title="Indicative savings (INR)", yaxis_title="")
            show_chart(fig)
        st.markdown("#### Purchase lines to review")
        st.caption("Sorted by largest combined landed-cost opportunity. Compare quoted vendor against the current supplier before negotiating or switching.")
        display_columns = ["purchase_order_id", "quarter", "material", "chemical_category", "quantity_kg", "vendor", "best_landed_vendor", "best_price_vendor", "best_freight_vendor", "source_country",
                           "price_per_kg_inr", "benchmark_price_per_kg_inr", "freight_cost_to_india_per_kg_inr",
                           "benchmark_freight_per_kg_inr", "benchmark_landed_per_kg_inr", "combined_saving_inr"]
        present_columns = [column for column in display_columns if column in savings.columns]
        opportunity_table = savings[savings["has_comparable_offer"]].sort_values("combined_saving_inr", ascending=False)[present_columns].head(250).rename(columns={
            "purchase_order_id": "PO", "quarter": "Quarter", "material": "Material", "chemical_category": "Category", "quantity_kg": "Qty (kg)",
            "vendor": "Current vendor", "best_landed_vendor": "Best landed vendor", "best_price_vendor": "Best price vendor",
            "best_freight_vendor": "Best freight vendor", "source_country": "Origin", "price_per_kg_inr": "Current price (₹/kg)",
            "benchmark_price_per_kg_inr": "Best price quote (₹/kg)", "freight_cost_to_india_per_kg_inr": "Current freight (₹/kg)",
            "benchmark_freight_per_kg_inr": "Best freight quote (₹/kg)", "benchmark_landed_per_kg_inr": "Best landed quote (₹/kg)",
            "combined_saving_inr": "Combined opportunity (₹)",
        })
        st.dataframe(opportunity_table, width="stretch", hide_index=True, height=390)
        st.download_button("Download opportunity lines (CSV)", data=opportunity_table.to_csv(index=False).encode("utf-8"),
                           file_name="purchase_savings_opportunities.csv", mime="text/csv")
    with st.expander("How the savings estimate works"):
        st.markdown(
            "- **Material price:** (current price/kg − lowest comparable quoted price/kg) × purchased kg, floored at zero.\n"
            "- **Freight:** (current freight/kg − lowest comparable quoted freight/kg) × purchased kg, floored at zero.\n"
            "- **Combined landed cost:** (current price + freight − lowest single quoted price + freight for the same material and origin) × kg, floored at zero.\n\n"
            "Offers are compared within the same material and country to avoid misleading cross-origin comparisons. The offer file contains indicative rates, not proof of availability or negotiated terms."
        )

st.divider()
with st.expander("Upload different data (optional)", expanded=False):
    st.caption("Upload both CSVs to replace the sample data. Clear both uploads to return to the included sample.")
    if bool(st.session_state.get("purchase_upload")) != bool(st.session_state.get("offer_upload")):
        st.warning("Upload both files for a like-for-like vendor and purchase review; the missing file currently uses sample data.")
    upload_columns = st.columns(2)
    with upload_columns[0]:
        st.file_uploader("Purchase history CSV", type="csv", key="purchase_upload")
    with upload_columns[1]:
        st.file_uploader("Vendor offers CSV", type="csv", key="vendor_upload")
st.caption(f"purchase_intelligence · Data reviewed: {quality['purchase_rows']:,} purchase rows · Analysis: {llm_review['source']} · Metrics reconciled in Python.")
