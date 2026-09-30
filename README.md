<div align="center">

# 🧪 purchase_intelligence

### A clearer view of chemical purchasing spend

Turn purchase and supplier quote files into understandable spend, supplier, and savings insights.

</div>

---

## 👋 What is this?

**purchase_intelligence** is a simple dashboard for a chemicals company. It helps category leaders, finance teams, and executives answer questions such as:

- How much did we spend on materials and freight?
- Is spend going up or down compared with recent quarters?
- Which categories and suppliers account for the most spend?
- Where might we be able to negotiate a better material price or freight rate?
- Are there unusual or incomplete data points that need checking?

The dashboard uses plain procurement language and Indian rupee values. No AI experience is needed to use it.

## 🚀 Start using the dashboard

### First time setup

Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

- **Linux or macOS:** `source .venv/bin/activate`
- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`

Install the required packages and start the dashboard:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in the terminal (usually `http://localhost:8501`).

### Choose a reporting quarter

Use **Reporting quarter** near the top of the page. The main spend measures, supplier view, category view, and savings opportunities update to that quarter.

- The **spend bars** compare the selected quarter with up to two earlier quarters, when available.
- The **landed-cost line chart** keeps the full history and highlights the selected quarter in orange.

### Use your own files

At the bottom of the dashboard, open **Upload different data (optional)** and provide both CSV files. Uploading only one file means the other input continues to use the sample data; the dashboard will show a reminder. Clear both uploads to return to the sample.

## 📊 What each tab shows

### Spend overview

A high-level picture of the selected quarter: landed spend, change from the prior quarter, average delivered cost per kg, and the freight share of spend. Charts show the recent spend comparison, the longer-term cost trend, category spend, and analysis checks.

### Vendor summary

A supplier scorecard focused on questions for procurement and finance leaders:

- How many suppliers were active?
- How much of spend is with the largest supplier and the top three suppliers?
- Which suppliers account for the most landed spend?
- Which suppliers have the largest indicative savings opportunities?

The horizontal bars are ranked so suppliers can be compared without interpreting bubble sizes.

### Savings opportunities

Shows three separate views of possible savings:

- **Material price:** the possible improvement if purchases matched the lowest comparable quoted material price.
- **Freight:** the possible improvement if purchases matched the lowest comparable quoted freight rate.
- **Combined landed cost:** the possible improvement if purchases matched the lowest single quote for total material price plus freight.

The tab also lists purchase lines to review and lets you download those opportunities as a CSV.

> **Important:** price-only and freight-only estimates can overlap. Do not add them together. The combined landed-cost opportunity is a separate comparison.

## 🧮 How the numbers work

For each valid purchase line:

- **Material spend** = quantity purchased × material price per kg
- **Freight spend** = quantity purchased × freight cost per kg
- **Landed spend** = material spend + freight spend
- **Average landed cost per kg** = total landed spend ÷ total quantity

Savings are compared only with offers for the **same material and source country**. This helps avoid misleading comparisons between different materials or origins.

The application calculates spend and savings with Python. The optional Groq AI service reviews quality findings and calculations and can provide a plain-English summary. Every AI-produced KPI is checked against an independent Python calculation; if it does not match, the dashboard uses the Python result instead. The AI does not get to change the underlying purchase arithmetic.

## 📁 The included sample data

Two reproducible CSV files are included so the dashboard can be explored without company data:

| File | What it contains |
| --- | --- |
| `data/purchase_history.csv` | Purchase lines across 10 quarters, from 2024 Q1 through 2026 Q2. Includes material, quantity, price, supplier, country, freight, chemical category, and family. |
| `data/vendor_offers.csv` | Vendor quotes by material and country, including quoted price per kg and freight to India per kg. |

The sample has **1,315 purchase lines** and **332 vendor offers**. Some purchase rates deliberately differ from the quote baseline so the data-quality and savings checks have examples to find. This is demonstration data, not actual company spend.

To recreate the same sample files:

```bash
python -m purchase_intelligence.data.generate_mock_data
```

The generator uses a fixed seed, so it recreates consistent sample data.

## 🧠 Optional Groq AI summary

The dashboard and all arithmetic work without a Groq key. To enable the AI review, add your key to a local `.env` file in the project folder:

```text
GROQ_API_KEY=your-key-here
```

You can optionally choose a Groq model with `GROQ_MODEL`. Keep `.env` private; never share it or commit it to source control. If Groq is unavailable, the app uses the local Python analysis and remains usable.

## ☁️ Deploy on Streamlit Community Cloud

The code is published in the GitHub repository **Purchase-Intelligence**, on the **`main`** branch. To deploy it:

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize Streamlit to access this repository. If the repository is private, also grant access to private repositories in the Streamlit account's **Settings → Linked accounts → Source control**.
2. Choose **Create app**, then select `pavandeep-godi/Purchase-Intelligence`, branch `main`, and the app file `app.py`.
3. If you want Groq AI on the hosted app, open the app's **Settings → Secrets** and add:

	```toml
	GROQ_API_KEY = "your-key-here"
	```

	Do not put the key in a GitHub file. The hosted app reads Streamlit Secrets; the local app can read the ignored `.env` file. The app can be deployed without a key, using Python analysis and local summaries.

If Streamlit says it cannot find or connect to the repository, first check that you selected the `main` branch and `app.py`, and that Streamlit has GitHub permission to access the repository. Repository admins can authorize the GitHub connection.

## 📝 Create a shareable analysis report

Generate a plain-text summary and a machine-readable JSON report from the included CSVs:

```bash
python run_analysis.py
```

Reports are saved in `reports/`:

- `purchase_summary.txt` — executive summary, quality findings, KPIs, and calculation reconciliation.
- `purchase_summary.json` — the same analysis in a structured format.

## ✅ Run the checks

To confirm the project calculations and data checks are working:

```bash
python -m unittest discover -s tests -v
```

## ⚠️ Use savings as leads, not promises

A lower quote does not automatically mean a supplier can provide the same quality, quantity, delivery schedule, or commercial terms. The estimates do not include supplier qualification, capacity, taxes or duties, foreign exchange, lead time, contract terms, or switching costs. Procurement, operations, and finance should verify each opportunity before taking action.
