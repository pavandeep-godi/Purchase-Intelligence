# Purchase Intelligence project guidance

- Keep analysis calculations in `purchase_intelligence/`; keep UI presentation in `app.py`.
- Maintain deterministic mock data generation with a fixed random seed and preserve the documented column names.
- Validate totals and savings calculations against row-level arithmetic before displaying them.
- The dashboard must work without a Groq key; Groq is an optional narrative enhancement only.
- Use clear procurement language, INR/kg labels, accessible chart titles, and concise explanations for business users.
- When changing dependencies or launch behavior, update `requirements.txt` and `README.md`.
