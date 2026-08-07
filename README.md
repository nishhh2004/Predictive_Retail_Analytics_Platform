# 🎯 Predictive Retail Analytics Platform
[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Access_App-00d4ff?style=for-the-badge&logo=streamlit&logoColor=white)](https://predictive-retail-analytics-platform.onrender.com/)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![SARIMA](https://img.shields.io/badge/Statsmodels_(SARIMA)-FF6F00?style=for-the-badge&logo=scipy&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

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
* **Interactive Dashboard:** Streamlit, Plotly, streamlit-option-menu

## 📊 Executive Interface

*(Below is a preview of the interactive Power BI application driving the financial and operational insights.)*

![Inventory Optimization Engine](images/Inventory%20Optimization.png)
*(Above: The Operational Engine tracking baseline metrics and unit demand.)*

![Financial Impact](images/Financial%20Impact.png)
*(Above: Scenario testing and financial risk breakdown by product category.)*

![Model Validation](images/Model%20Validation.png)
*(Above: Visual validation of the statistical forecasting models.)*

## 📂 Project Structure

```text
├── app.py                        # Streamlit analytics dashboard
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

## 🖥️ Interactive Streamlit Dashboard

The platform includes a premium, multi-page **Streamlit** web application with three modules:

| Page | Description |
|---|---|
| **Executive Dashboard** | KPI cards, historical vs. predicted sales chart, inventory risk breakdown |
| **Demand Forecasting** | Interactive SARIMA forecast interface with configurable store, category, and horizon |
| **Inventory Optimization** | Reorder recommendation engine with action-coded SKU table |

Launch the dashboard:
```bash
streamlit run app.py
```

## ⚙️ How to Run the Project

**1.Clone this repository:**
```
git clone https://github.com/nishhh2004/Predictive_Retail_Analytics_Platform.git
```

**2.Install the required Python libraries:**
```
pip install -r requirements.txt
```

3. **Data Setup:** 
   * Most of the foundational data (e.g., `oil.csv`, `stores.csv`, `holidays_events.csv`) is already included in the `data/raw/` directory of this repository.
   * Due to GitHub's file size limits, the two largest datasets must be downloaded externally from this [Google Drive Link](https://drive.google.com/drive/folders/1otuq3_I0g40NaEekUTqaYHOstFUemTX6?usp=sharing).
   * Once downloaded, place `master_sales_data.csv` directly into the `data/` directory.
   * Place the `train.csv` file into the `data/raw/` directory.

**4.Notebooks:** Open the Jupyter Notebooks in sequential order (01 -> 02 -> 03 -> 04) to view the data pipeline, exploratory analysis, and statistical modeling.

**5.Power BI Dashboard:** Open the Inventory_Optimization_App.pbix file in Power BI Desktop to interact with the final visualizations.

**6.Streamlit Dashboard:** Run `streamlit run app.py` to launch the interactive web application.

## 👨‍💻 Author

**Nishanth**
* **LinkedIn:** [Connect with me on LinkedIn](https://www.linkedin.com/in/nishanth-ms)
* **GitHub:** [github.com/nishhh2004](https://github.com/nishhh2004)
