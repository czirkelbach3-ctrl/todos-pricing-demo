# AI Retail Pricing Optimizer — Prototype

Interactive Streamlit app that recommends dynamic prices for retail SKUs
(dairy and snacks) based on demand signals, competitor prices, time-of-day
factors, and own-price elasticity. Compares an AI strategy against a static
(everyday) pricing baseline.

## Project structure

```
.
├── app.py                  # Streamlit app
├── data/
│   └── sample_skus.csv     # Synthetic SKU master (Kaggle-style schema)
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

## Deploy as a shareable demo (Streamlit Community Cloud)

Streamlit Community Cloud gives you a free public URL. Steps:

1. Push this folder to a public GitHub repo (root must contain `app.py` and
   `requirements.txt`).
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **New app**, pick the repo, branch, and `app.py` as the entry point.
4. Click **Deploy**. You'll get a `https://<your-app>.streamlit.app` URL you
   can share.

Alternative one-click hosts: **Hugging Face Spaces** (Streamlit SDK) and
**Render**. Both accept the same `requirements.txt` + `app.py` layout.

## Swap in real Kaggle data

Replace `data/sample_skus.csv` with any file having these columns:

| column              | description                                     |
| ------------------- | ----------------------------------------------- |
| `sku_id`            | unique identifier                               |
| `sku_name`          | display name                                    |
| `category`          | e.g. Dairy, Snacks                              |
| `unit_cost`         | cost of goods (USD)                             |
| `static_price`      | current/everyday shelf price                    |
| `base_demand_units` | typical units sold per period at static price   |
| `price_elasticity`  | own-price elasticity (negative number)          |
| `competitor_price`  | observed competitor price                       |
| `shelf_life_days`   | shelf life (used for perishability messaging)   |

Good public datasets to mine for these fields:

- **Retail Price Optimization** by Sandeep Singh (Kaggle) — price/units
  panels you can regress for elasticity.
- **Grocery Store Dataset** by Heeral Dedhia (Kaggle) — SKU master fields.
- **Instacart Market Basket Analysis** (Kaggle) — demand baselines.

A standard way to estimate elasticity from those datasets is to fit
`log(units) ~ log(price) + promo + week + sku_fixed_effect` and read the
coefficient on `log(price)` per SKU or category.

## Pricing methodology (one paragraph)

We start from the textbook profit-maximizing markup under constant
elasticity, `P* = c·ε/(ε+1)`, blend it with the observed competitor price
using a user-controlled anchoring weight, then apply a capped multiplicative
tilt for demand signals and time-of-day, and finally clamp the result to a
minimum-margin floor and a maximum allowed move from the static price. The
app reports forecast units, revenue, and margin under both the static and
AI strategies so you can see the trade-off directly.
