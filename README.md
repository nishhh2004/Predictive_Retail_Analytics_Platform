# 🎯 Predictive Retail Analytics Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![SARIMA](https://img.shields.io/badge/Statsmodels_(SARIMA)-FF6F00?style=for-the-badge&logo=scipy&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

## 📌 Project Overview

**Predictive Retail Analytics Platform** is an end-to-end data pipeline that bridges raw database extraction with advanced statistical forecasting and executive-level visualization. This project analyzes retail records to optimize inventory strategy, forecast demand, and test business scenarios using synthetic and historical data.

Unlike standard visualization projects, this repository demonstrates a complete lifecycle: from data collection and validation to predictive statistical modeling and operational UI design.

The project is structured into four core components:

1. **Data Engineering (SQL & Python):** Extracting, cleaning, and structuring raw transactional data (including external factors like oil prices and holiday events) for time-series analysis.
2. **Exploratory Data Analysis (EDA):** Identifying macro-environmental spending trends, seasonal spikes, and baseline operational metrics.
3. **Statistical Forecasting:** Engineering autoregressive statistical models to mathematically predict future inventory demand.
4. **Inventory Optimization & Visualization (Power BI):** Generating metrics to simulate business stress-tests, visualized through a premium, interactive UI for executive decision-making.

## 🛠️ Tech Stack & Tools

* **Languages:** Python (Pandas, NumPy), SQL
* **Statistical Modeling:** `statsmodels` (SARIMA), `scikit-learn` (preprocessing)
* **Data Visualization:** Matplotlib, Seaborn
* **Business Intelligence:** Power BI (Custom UI/UX formatting)

## 📊 Executive Interface

*(Below is a preview of the interactive Power BI application driving the financial and operational insights.)*

![Inventory Optimization Engine](images/inventory_optimization.png)
*(Above: The Operational Engine tracking baseline metrics and unit demand.)*

![Financial Impact](images/financial_impact.png)
*(Above: Scenario testing and financial risk breakdown by product category.)*

![Model Validation](images/model_validation.png)
*(Above: Visual validation of the statistical forecasting models.)*

## 📂 Project Structure

```text
├── dashboards/
│   └── Inventory_Optimization_App.pbix
├── data/
│   ├── master_sales_data.csv
│   └── raw/
│       ├── holidays_events.csv
│       ├── oil.csv
│       ├── stores.csv
│       ├── test.csv
│       ├── train.csv
│       └── transactions.csv
├── images/
│   ├── financial_impact.png
│   ├── inventory_optimization.png
│   └── model_validation.png
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_forecasting.ipynb
│   └── 04_inventory_optimization.ipynb
├── outputs/
│   ├── correlation_heatmap.png
│   ├── daily_sales_trend.png
│   ├── forecast_results.csv
│   ├── forecast_results.png
│   ├── inventory_metrics.csv
│   ├── inventory_optimization.csv
│   ├── promo_effectiveness.png
│   ├── seasonal_decomposition.png
│   ├── top_families.png
│   └── train_test_split.png
├── sql/
│   └── analysis_queries.sql
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ How to Run the Project

1.Clone this repository:
```
git clone [https://github.com/nishhh2004/Predictive_Retail_Platform.git](https://github.com/nishhh2004/Predictive_Retail_Platform.git)
```

2.Install the required Python libraries:
```
pip install -r requirements.txt
```

3.Data Setup: Ensure the raw CSV files are located in the data/raw/ directory.

4.Notebooks: Open the Jupyter Notebooks in sequential order (01 -> 02 -> 03 -> 04) to view the data pipeline, exploratory analysis, and statistical modeling.

5.Dashboard: Open the Inventory_Optimization_App.pbix file in Power BI Desktop to interact with the final visualizations.

## 👨‍💻 Author

**Nishanth**
* **LinkedIn:** [Connect with me on LinkedIn](https://www.linkedin.com/in/nishanth-ms)
* **GitHub:** [github.com/nishhh2004](https://github.com/nishhh2004)
