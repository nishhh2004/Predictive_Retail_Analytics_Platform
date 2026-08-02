# 📊 Predictive Retail Analytics Platform — Complete Project Report

> **Purpose**: If you read this once, you can confidently explain every part of this project to anyone — a recruiter, a hiring manager, a data science interviewer, or a technical panel. Everything is explained in plain English first, then with the technical details.

---

## 1. What Is This Project? (The 30-Second Pitch)

**One-liner**: *"I built an end-to-end data pipeline that takes 3 million+ raw retail sales records, cleans and consolidates them, trains a SARIMA time-series model to forecast daily demand, and then uses those forecasts to recommend exactly how much inventory each product category should carry — all visualized in a premium interactive web dashboard."*

**Business problem**: A retail chain (Corporación Favorita, Ecuador's largest grocery retailer) has 54 stores selling 33 product categories. They need to answer:
- *"How many units of GROCERY will we sell next week?"*
- *"Should we reorder BEVERAGES now, or can we wait?"*
- *"Where is our biggest financial risk — stockouts (running out) or overstocking (having too much)?"*

**What I built**:
1. A **SQL + Python data pipeline** that merges 6 raw datasets into one clean master table
2. A **SARIMA forecasting model** that predicts future daily sales with 92.1% accuracy (7.9% error)
3. An **inventory optimization engine** that calculates safety stock, reorder points, and economic order quantities
4. A **Streamlit web application** with 3 interactive pages for executives to make data-driven decisions

---

## 2. The Data — Where It Comes From

### 2.1 The 6 Raw Datasets

| Dataset | Records | What It Contains | Why It Matters |
|---|---|---|---|
| **train.csv** | ~3M rows | Daily sales per store per product family (2013–2017) | The core — this is what we're predicting |
| **test.csv** | ~28K rows | Same structure, but without sales values | Used for Kaggle submission / validation |
| **stores.csv** | 54 rows | Store metadata: city, state, type (A/B/C/D/E), cluster | Lets us segment by geography and store type |
| **oil.csv** | ~1,200 rows | Daily crude oil price (WTI) | Ecuador's economy is oil-dependent; oil price drops → consumer spending drops |
| **holidays_events.csv** | ~350 rows | National/regional/local holidays and events | Holidays cause demand spikes (Christmas, Carnival, etc.) |
| **transactions.csv** | ~83K rows | Daily transaction count per store | Proxy for foot traffic; validates sales patterns |

### 2.2 How I Combined Them (The Master Table)

Using SQL JOINs (see `sql/analysis_queries.sql`, Query Q4), I merged everything into `master_sales.csv` (~266 MB):

```
sales ← LEFT JOIN stores    (on store_nbr)
       ← LEFT JOIN oil       (on date)
       ← LEFT JOIN transactions (on date + store_nbr)
       ← LEFT JOIN holidays   (on date)
```

**Key data cleaning steps**:
- Oil prices were missing on weekends → **forward-filled** (used last known Friday price for Saturday/Sunday)
- Some sales were negative (returns) → kept as-is since they reflect real business patterns
- Missing transaction counts → left as NULL (not all stores reported daily)

### 2.3 Scale of the Data

- **3,000,888** sales records
- **54** stores × **33** product families × **~1,684** days
- Date range: **January 1, 2013 → August 15, 2017** (4.5 years)

---

## 3. Exploratory Data Analysis (EDA) — What the Data Told Us

### 3.1 Key Findings (from Notebook 02_EDA)

| Finding | What It Means |
|---|---|
| **Strong weekly seasonality** | Sales spike on weekends (Saturday/Sunday are ~30% higher than Tuesday/Wednesday) |
| **Monthly patterns** | Month-end and month-start show higher sales (payday effect) |
| **Year-over-year growth** | Sales increased ~8% annually from 2013 to 2017 |
| **Oil price correlation** | When oil prices dropped 50% (2014–2015), consumer spending also fell — Ecuador's economy depends on oil exports |
| **Holiday spikes** | Christmas week sales are 2–3× normal; Carnival and Easter also show clear spikes |
| **Top 3 families dominate** | GROCERY I (27%), BEVERAGES (17%), and PRODUCE (10%) account for over half of all sales |
| **Promotions work** | Products on promotion sell on average **40–70% more** than when not promoted |

### 3.2 Stationarity Check (Why It Matters for SARIMA)

- **ADF Test on raw series**: p-value > 0.05 → **not stationary** (the data has a trend)
- **ADF Test after differencing (d=1)**: p-value < 0.05 → **now stationary** ✅
- This told us the model needs **d=1** (one level of differencing to remove the trend)

---

## 4. The SARIMA Model — How the Forecasting Works

### 4.1 What Is SARIMA? (Plain English)

SARIMA stands for **Seasonal Auto-Regressive Integrated Moving Average**. Think of it as a mathematical formula that says:

> *"To predict tomorrow's sales, I'll look at: (1) how sales have been trending recently, (2) the pattern from the same day last week, and (3) how far off my recent predictions were — and adjust."*

It has 7 parameters: **(p, d, q) × (P, D, Q, m)**

| Parameter | What It Does | Plain English |
|---|---|---|
| **p** (AR order) | How many recent days to look back | "Yesterday and the day before influence today" |
| **d** (Differencing) | How many times to subtract consecutive values | "Remove the overall trend so we can see the pattern" |
| **q** (MA order) | How many recent prediction errors to factor in | "If I was wrong yesterday, adjust today's prediction" |
| **P** (Seasonal AR) | Same as p, but for the seasonal cycle | "Same day last week matters" |
| **D** (Seasonal differencing) | Seasonal trend removal | "Remove the weekly pattern's trend" |
| **Q** (Seasonal MA) | Seasonal error correction | "If I was wrong last Saturday, adjust this Saturday" |
| **m** (Season length) | Length of one cycle | **m=7** because retail has a weekly cycle |

### 4.2 How I Found the Best Parameters

I used a **grid search** — tried every combination of:
- p ∈ {0, 1, 2}, d ∈ {0, 1}, q ∈ {0, 1, 2}
- P ∈ {0, 1}, D ∈ {1}, Q ∈ {0, 1}
- m = 7 (weekly seasonality)

That's **108 combinations** tested. Each one trains a model and measures **AIC (Akaike Information Criterion)** — lower = better fit without overfitting.

- Used last 365 days of training data for the search (to keep it computationally feasible)
- Then retrained the winning model on the **full training set**

### 4.3 Train/Test Split

| Set | Date Range | Days | Purpose |
|---|---|---|---|
| **Train** | 2013-01-01 → 2017-07-16 | ~1,654 | Model learns patterns from this |
| **Test** | 2017-07-17 → 2017-08-15 | 30 | Model has never seen this — we measure accuracy here |

### 4.4 Model Performance — The 7.9% MAPE

The model achieved **MAPE = 7.9%**, meaning on average, the prediction is off by 7.9% from the actual value.

**What that means in real terms**: If actual sales on a day were 800,000 units, the model's prediction would typically be between **737,000 and 863,000** units.

| Metric | Value | What It Means |
|---|---|---|
| **MAPE** | 7.9% | Average percentage error — lower is better |
| **MAE** | ~63,000 units | Average absolute error in units |
| **Accuracy** | 92.1% | 100% - MAPE = how often the model is "right" |

**How MAPE is calculated**:
```
MAPE = (1/n) × Σ |actual - forecast| / |actual| × 100

For each day:
  error_pct = |actual_sales - forecasted_sales| / actual_sales × 100

Then average all the daily error percentages.
```

### 4.5 Confidence Intervals (The Shaded Band)

The model also outputs **95% confidence intervals**:
- Upper bound = forecast + 1.96 × standard error
- Lower bound = forecast − 1.96 × standard error
- "We're 95% confident the actual value will fall within this band"

---

## 5. Inventory Optimization — The Business Impact

### 5.1 The Three Key Metrics

#### 🔹 Safety Stock (1.38M units)

**Plain English**: "How much extra inventory should you keep as a buffer in case demand is higher than expected?"

**Formula**:
```
Safety Stock = Z × σ × √(Lead Time)

Where:
  Z = 1.65 (for 95% service level — only 5% chance of stockout)
  σ = standard deviation of daily demand (how much demand varies day-to-day)
  Lead Time = 7 days (assumed time for a new shipment to arrive)
```

**Example for GROCERY I**:
- σ (std of daily demand) = 77,318 units
- Safety Stock = 1.65 × 77,318 × √7 = **337,531 units**
- This means: "Keep 337K extra units of GROCERY I in the warehouse as a buffer"

**Total across all 33 families = 1.38M units**

#### 🔹 Reorder Point (When to Place an Order)

**Plain English**: "When your stock drops to this number, place a new order immediately."

**Formula**:
```
Reorder Point = (Avg Daily Demand × Lead Time) + Safety Stock

For GROCERY I:
  = (208,330 × 7) + 337,531 = 1,795,839 units
```

If GROCERY I stock drops below ~1.8M units, the system flags: "ORDER NOW."

#### 🔹 Inventory Risk ($4.61B)

**Plain English**: "How much money has the company lost (or could lose) due to bad inventory management?"

This combines two types of cost:

| Risk Type | What Happens | Cost Per Unit | How It's Calculated |
|---|---|---|---|
| **Stockout cost** | Demand exceeded what we had → lost sales | $15/unit | Rolling 7-day demand > reorder point → those excess units × $15 |
| **Overstock cost** | We had way more than needed → sitting in warehouse | $2/unit | Rolling 7-day demand < 50% of expected → excess units × $2 |

**Total inventory risk = stockout costs + overstock costs across all families = $4.61B**

This is the total **historical** cost exposure over the ~4.5-year dataset. It answers: "If we had used optimal inventory management, how much could we have saved?"

#### 🔹 EOQ (Economic Order Quantity)

**Plain English**: "When you do reorder, how many units should you order at once?"

**Formula**:
```
EOQ = √(2 × Annual Demand × Ordering Cost / Holding Cost)

Assumptions:
  Ordering Cost = $50 per order (admin, shipping, processing)
  Holding Cost = 25% of unit cost per year
  Unit Cost = $10

For GROCERY I:
  Annual Demand = 208,330 × 365 = 76,040,359
  EOQ = √(2 × 76,040,359 × 50 / 2.50) = 55,151 units per order
```

---

## 6. The Web Application — Page by Page

### 6.1 Architecture

```
Streamlit (Python)
├── app.py (single file, ~1,050 lines)
├── Data: outputs/*.csv (pre-computed by notebooks)
├── UI: Custom CSS injected via st.markdown
├── Charts: Plotly (interactive, dark-themed)
└── Navigation: streamlit-option-menu (sidebar)
```

**Why Streamlit?** It's a Python-native framework that turns data scripts into web apps with zero JavaScript. Perfect for data science projects where the backend is already Python.

### 6.2 Page 1: Executive Dashboard

**Who it's for**: C-suite executives, VPs who need a 30-second snapshot.

| Element | What It Shows | Data Source |
|---|---|---|
| **KPI: Overall MAPE 7.9%** | Model accuracy — "Our forecasts are 92.1% accurate" | Hardcoded (computed in notebook 03) |
| **KPI: Total Safety Stock 1.38M** | Sum of recommended buffer stock across all 33 families | `inventory_metrics.csv` → sum of `safety_stock` column |
| **KPI: Inventory Risk $4.61B** | Total stockout + overstock cost exposure | `inventory_simulation.csv` → sum of `total_inventory_cost` |
| **KPI: Avg Daily Demand 651K** | Total units sold per day on average across all families | `inventory_metrics.csv` → sum of `avg_daily_demand` |
| **Line Chart** | Actual sales (teal) vs SARIMA forecast (blue dotted) with 95% CI band | `forecast_results.csv` |
| **Donut Chart** | Which product families contribute most to inventory risk | `inventory_simulation.csv` → top 8 by `total_inventory_cost` |
| **Error Bar Chart** | Daily forecast error %, color-coded (green < 5%, amber 5-10%, red > 10%) | `forecast_results.csv` → `error_pct` column |

### 6.3 Page 2: Demand Forecasting Engine

**Who it's for**: Supply chain analysts, demand planners who need to run scenarios.

| Element | What It Does |
|---|---|
| **Store ID dropdown** | Select any of the 54 Favorita stores |
| **Product Category dropdown** | Select any of the 33 product families |
| **Date picker** | Choose when the forecast should start |
| **Horizon slider** | Choose how many days ahead to forecast (7–90) |
| **Holiday toggle** | ON = include holiday demand spikes in the forecast |
| **Oil price toggle** | ON = factor in oil price effects on consumer spending |
| **KPIs** | Avg forecast, peak demand, historical baseline, demand shift % |
| **Forecast chart** | Blue = historical (90 days), Teal = forecast, Green band = 95% CI |
| **Vertical line** | Marks where historical data ends and forecast begins |

**Important note for interviews**: The forecast on this page is currently **mocked** using numpy (sinusoidal seasonal patterns + trend + noise). The code has clearly marked `# PLACEHOLDER` comments showing exactly where to plug in the real `.pkl` SARIMA model. This was a deliberate design decision because running live SARIMA inference on each user interaction would be too slow for a demo — the real model takes minutes to train.

### 6.4 Page 3: Inventory Optimization Engine

**Who it's for**: Warehouse managers, procurement teams who decide what to order.

| Element | What It Shows |
|---|---|
| **KPI: Urgent Reorders** | Count of SKUs where current stock < 40% of reorder point |
| **KPI: Standard Reorders** | Count of SKUs where stock is 40–85% of reorder point |
| **KPI: Stock Healthy** | Count of SKUs above reorder threshold |
| **KPI: Total Reorder Volume** | Sum of EOQ units that need ordering |
| **Recommendation Table** | Each row = one product family with: SKU code, current stock, 30-day demand forecast, safety stock, reorder point, EOQ, and action badge (🔴/🟡/🟢) |
| **Bar Chart** | Current stock (blue) vs reorder point (red) per family — visual gap analysis |
| **Donut Chart** | Distribution of actions (how many urgent vs reorder vs hold) |

**Action logic**:
```
stock_ratio = current_stock / reorder_point

if ratio < 0.40 → 🔴 URGENT REORDER (critically low)
if ratio < 0.85 → 🟡 REORDER (approaching danger)
else            → 🟢 HOLD (stock is healthy)
```

---

## 7. The SQL Queries — What I Demonstrated

The `sql/analysis_queries.sql` file contains 10 production-ready SQL queries demonstrating:

| Query | Skill Demonstrated |
|---|---|
| Q1: Sales by family | `GROUP BY`, `ORDER BY`, aggregation |
| Q2: Monthly sales trend | `DATE_FORMAT()`, time-series aggregation |
| Q3: Sales by store type | Multi-table `JOIN` |
| Q4: Master JOIN | 4-table `LEFT JOIN` (the data pipeline core) |
| Q5: Holiday impact | `JOIN` with external events, `AVG()` |
| Q6: Sales by state | `COUNT(DISTINCT)`, derived metrics |
| Q7: Rolling averages | **Window Functions** (`AVG() OVER`, `ROWS BETWEEN`) |
| Q8: Year-over-year growth | **`LAG()` window function**, `NULLIF`, YoY calculation |
| Q9: Top stores per state | **`RANK() OVER (PARTITION BY)`** |
| Q10: Promo effectiveness | **Conditional aggregation** (`CASE WHEN` inside `AVG`) |

---

## 8. Tech Stack Summary

| Layer | Technology | Why I Chose It |
|---|---|---|
| Data Engineering | Python (Pandas, NumPy), SQL | Industry standard for data pipelines |
| Statistical Modeling | statsmodels (SARIMA) | Best library for time-series in Python; gives confidence intervals |
| Parameter Tuning | scikit-learn, custom grid search | Systematic approach to finding optimal model |
| Visualization (static) | Matplotlib, Seaborn | For notebooks and static output images |
| Visualization (interactive) | Plotly | Interactive, web-ready charts with hover tooltips |
| Web Framework | Streamlit | Python-native, zero JS, perfect for data apps |
| Navigation | streamlit-option-menu | Sleek sidebar nav with icons |
| BI Dashboard | Power BI | Executive-level formatting (separate from Streamlit) |
| Version Control | Git, GitHub | Standard collaboration and deployment |

---

## 9. Project Structure Explained

```
Predictive_Retail_Analytics_Platform/
│
├── app.py                          ← The Streamlit web application (this report covers it)
│
├── notebooks/
│   ├── 01_data_check.ipynb         ← Validate raw CSVs, check shapes/nulls/dtypes
│   ├── 02_EDA.ipynb                ← Exploratory analysis, visualizations, correlations
│   ├── 03_forecasting.ipynb        ← SARIMA model: stationarity, grid search, train, evaluate
│   └── 04_inventory_optimization.ipynb ← Safety stock, reorder points, EOQ, cost simulation
│
├── data/
│   ├── master_sales.csv            ← 266MB merged dataset (created by notebook 01)
│   └── raw/                        ← 6 original CSVs from Kaggle
│
├── outputs/
│   ├── forecast_results.csv        ← 30-day actual vs predicted (from notebook 03)
│   ├── inventory_metrics.csv       ← Per-family safety stock, reorder point, EOQ
│   ├── inventory_simulation.csv    ← Stockout/overstock cost simulation
│   └── *.png                       ← Static chart exports
│
├── sql/
│   └── analysis_queries.sql        ← 10 SQL queries demonstrating advanced skills
│
├── dashboards/
│   └── Inventory_Optimization_App.pbix  ← Power BI dashboard
│
├── requirements.txt                ← All Python dependencies
└── README.md                       ← Project documentation
```

---

## 10. Common Interview Questions & Answers

### General / Behavioral

**Q: Walk me through this project.**
> "I started with 6 raw retail datasets totaling 3M+ records from Ecuador's largest grocery chain. I built a SQL data pipeline to merge them into a single master table, then ran exploratory analysis to find weekly/monthly seasonality and oil price correlations. I trained a SARIMA model that forecasts daily demand with 7.9% MAPE. Then I built an inventory optimization engine that calculates safety stock, reorder points, and identified $4.61B in inventory risk. Finally, I built a Streamlit web app so stakeholders can interact with the forecasts and reorder recommendations."

**Q: What was the hardest part?**
> "Two things: (1) The grid search for SARIMA parameters — 108 combinations on 3M records was computationally expensive, so I used a 365-day subset for tuning, then retrained the winner on full data. (2) Oil price data was missing on weekends, which broke the time-series alignment until I implemented forward-fill imputation."

**Q: Why SARIMA and not Prophet / LSTM / XGBoost?**
> "SARIMA was the right choice because: (1) the data has clear, regular seasonality (weekly), which SARIMA handles natively with the seasonal component; (2) it gives built-in confidence intervals, which are critical for inventory planning; (3) it's interpretable — I can explain each parameter to a business stakeholder. Prophet would be a good alternative but abstracts away the statistical rigor. LSTMs are overkill for univariate daily data with regular seasonality."

### Technical — Model

**Q: How did you validate stationarity?**
> "I used the Augmented Dickey-Fuller (ADF) test. The raw series was non-stationary (p > 0.05), so I differenced it once (d=1), which made it stationary (p < 0.05). This told me the model needs first-order differencing."

**Q: What does m=7 mean?**
> "It's the seasonal period. Retail sales follow a weekly cycle — Saturdays are always high, Tuesdays are always low. So m=7 tells the model 'the pattern repeats every 7 days.'"

**Q: What's the difference between AIC and MAPE?**
> "AIC (Akaike Information Criterion) is used during model selection — it balances fit quality against model complexity (penalizes too many parameters). MAPE is the final evaluation metric — it tells us how accurate the chosen model's predictions are in percentage terms."

**Q: What would you do to improve the model?**
> "Three things: (1) Add exogenous variables (oil price, promotions, holidays) using SARIMAX instead of SARIMA; (2) Train separate models per store or per product family instead of aggregating everything; (3) Ensemble with a gradient-boosted model (XGBoost) for non-linear patterns."

### Technical — Inventory

**Q: Why Z = 1.65?**
> "It corresponds to a 95% service level from the standard normal distribution. This means we accept a 5% probability of stockout during lead time. If the business wants 99% service level, we'd use Z = 2.33, which would increase safety stock."

**Q: What's the relationship between safety stock and service level?**
> "They're directly proportional. Higher service level → higher Z-score → more safety stock → more holding cost. It's a business trade-off: spending more on warehouse space vs. risking lost sales."

**Q: How is the $4.61B calculated?**
> "For each product family, I ran a rolling 7-day demand window across the entire 4.5-year history. Any period where demand exceeded the reorder point = stockout (costed at $15/unit lost sale). Any period where demand was less than 50% of expected = overstock (costed at $2/unit holding). Sum everything up = $4.61B in total inventory cost exposure."

### Technical — Web App

**Q: Why Streamlit instead of Flask/Django?**
> "Streamlit is purpose-built for data applications. It's 10× faster to build than Flask because there's no HTML/JS to write — everything is Python. It has built-in caching (`@st.cache_data`), interactive widgets, and DataFrame rendering. For a data science project where the backend is already Python, it's the optimal choice."

**Q: How does the caching work?**
> "I use `@st.cache_data(ttl=600)` on data-loading functions. Streamlit hashes the function inputs and stores the output. For 600 seconds (10 minutes), subsequent calls with the same inputs return the cached result instantly, avoiding re-reading CSVs on every page interaction."

**Q: Is the forecast on the Demand Forecasting page real?**
> "The Executive Dashboard uses real model output from `forecast_results.csv`. The interactive Demand Forecasting page uses a mock generator (numpy-generated seasonal patterns) because running live SARIMA inference per interaction would take minutes. The code has clearly marked placeholder comments showing exactly where to plug in the real model — it's designed to be production-swappable."

**Q: How would you deploy this?**
> "Three options: (1) Streamlit Community Cloud (free, just connect the GitHub repo); (2) Docker container → AWS ECS or Google Cloud Run; (3) Heroku with a Procfile. For production, I'd add authentication, a PostgreSQL database instead of CSVs, and a scheduled model retraining pipeline."

### Data Engineering

**Q: Why did you use LEFT JOINs instead of INNER JOINs?**
> "Because oil prices and transactions have missing dates (weekends, holidays). An INNER JOIN would drop sales records where oil data is unavailable, losing valid rows. LEFT JOIN preserves all sales records and fills unmatched columns with NULL, which I then handle via forward-fill."

**Q: How would you handle this at scale?**
> "For 3M rows, Pandas works fine. At 100M+ rows, I'd use: (1) Apache Spark for distributed processing; (2) A data warehouse (BigQuery/Snowflake) with dbt for transformations; (3) Airflow for pipeline orchestration; (4) Delta Lake for versioned storage."

---

## 11. Numbers to Remember

| Metric | Value | Context |
|---|---|---|
| Raw records | 3,000,888 | Sales transactions across all stores/families |
| Raw datasets | 6 | train, test, stores, oil, holidays, transactions |
| Stores | 54 | Across Ecuador |
| Product families | 33 | GROCERY I, BEVERAGES, PRODUCE, etc. |
| Date range | 4.5 years | Jan 2013 – Aug 2017 |
| Model | SARIMA | (p,d,q)×(P,D,Q,7) with grid search |
| MAPE | 7.9% | = 92.1% accuracy |
| Safety stock | 1.38M units | Recommended buffer across all families |
| Inventory risk | $4.61B | Stockout + overstock cost exposure |
| Stockout cost | $15/unit | Lost sale penalty |
| Overstock cost | $2/unit | Holding cost penalty |
| Service level | 95% | Z = 1.65 |
| Lead time | 7 days | Assumed replenishment time |
| Web app pages | 3 | Dashboard, Forecasting, Optimization |
| SQL queries | 10 | Demonstrating JOINs, window functions, CTEs |
