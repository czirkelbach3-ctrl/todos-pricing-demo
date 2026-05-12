# Data folder

This folder holds Kaggle CSVs. **You do not need any files here to run the app** —
when the folder is empty, the app falls back to Demo Mode (synthetic data).

## Supported datasets

The app auto-detects either of these. If both are present, Favorita wins.

### Option A — Store Item Demand Forecasting Challenge
URL: https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

Place `train.csv` here:

```
data/train.csv
```

Columns expected: `date`, `store`, `item`, `sales`.

### Option B — Store Sales (Favorita)
URL: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Place these files here (rename `train.csv` → `favorita_train.csv` to avoid a
collision with Option A):

```
data/favorita_train.csv
data/stores.csv          (optional)
data/oil.csv             (optional)
data/holidays_events.csv (optional)
```

Columns expected on `favorita_train.csv`: `date`, `store_nbr`, `family`, `sales`,
`onpromotion`.

## How to download

Two equivalent options.

### Option 1 — manual upload
1. Sign in to Kaggle, accept the competition rules.
2. Download the CSVs and drop them into this folder.

### Option 2 — Kaggle CLI
```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place your kaggle.json API key at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Store Item Demand Forecasting
kaggle competitions download -c demand-forecasting-kernels-only -p data
unzip data/demand-forecasting-kernels-only.zip -d data

# Or Favorita
kaggle competitions download -c store-sales-time-series-forecasting -p data
unzip data/store-sales-time-series-forecasting.zip -d data
mv data/train.csv data/favorita_train.csv
```

## What the app does with the data

These public datasets contain only demand/sales — no prices, categories, vendor
attributes, promotions, or inventory. The app **enriches** them with synthetic
merchandising metadata (`src/data_loader.py::_enrich_with_synth_metadata`) so
every UI screen still has the fields it needs. Real production deployments
would replace the synthetic enrichment with the EDW item-master, vendor master,
and promo calendar.
