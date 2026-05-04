# Forecasting Electricity Demand in Nordic Power Systems

## 📌 Overview
Accurate electricity demand forecasting is essential for power system planning and electricity market operations. As weather patterns become more volatile due to climate change, understanding how temperature affects demand under different conditions has become increasingly important.

This project analyzes and forecasts daily electricity demand in **Norway, Sweden, and Finland**, focusing on how **temperature and demand volatility influence model performance**.

📄 [Read the full working paper on SSRN](https://dx.doi.org/10.2139/ssrn.6603718)

---

## ⚙️ Methodology

Two main approaches are compared:

- **SARIMAX** (Seasonal ARIMA with exogenous variables)  
  → Structured, interpretable, captures temporal dynamics  

- **XGBoost** (Extreme Gradient Boosting)  
  → Flexible, nonlinear machine learning model  

---

## 🌡 Key Idea

Forecast performance is evaluated under **different demand regimes**, defined by:

- **Demand volatility** (based on large day-to-day changes)
- **Temperature-driven vs non-temperature-driven periods**

---

## 📊 Evaluation

Model performance is assessed using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Seasonal and monthly analysis
- Volatility-based regime analysis
- **Diebold-Mariano tests** for statistical comparison

---

## 🔍 Key Findings

- Including **weather variables significantly improves forecast accuracy**
- **SARIMAX performs better** during temperature-driven volatility
- **XGBoost performs better** when volatility is not temperature-driven
- In Finland, where temperature sensitivity is weaker, model differences are smaller

👉 Overall:
> Forecast performance depends on how model structure interacts with temperature and demand regimes.

---

## 🌍 Relevance

These findings are important for:

- Managing **weather-related risks** in electricity systems  
- Improving **forecast reliability under climate variability**  
- Supporting **energy market operations and planning**

---

## 📁 Project Structure

- [data/](./data)  
  Raw dataset (not included due to ENTSO-E data access restrictions).

- [gee/](./gee)  
  Google Earth Engine scripts used to extract Population-Weighted ERA5-Land weather data.

- [notebooks/](./notebooks)  
  End-to-end analysis workflow: data preparation, modeling, and evaluation.

- [src/](./src)  
  Modular Python utilities for:
  - data processing  
  - SARIMAX models  
  - XGBoost models  
  - evaluation and metrics  

- [outputs/](./outputs)  
  Final figures and tables used in the analysis.

---

## 🌍 Climate Change Visualization Tool

The `scripts/` folder includes an interactive script that allows users to compare seasonal temperature patterns across different regions using **Google Earth Engine (GEE)**.

### Default comparison:
- **North:** Norway, Sweden, Finland  
- **South:** Spain, Portugal  
- **Years:** 1990 vs 2025  

### Requirements

Users must create their own Google Earth Engine account:

👉 https://earthengine.google.com/

Authenticate locally:

```bash
earthengine authenticate
```

---

<p align="center">
  <img src="outputs/figures/north_south_seasonal_temperature_1990_2025.png" width="850"/>
</p>

<p align="center">
  <em>Figure 1: Seasonal mean temperature comparison between Northern Europe and Southern Europe.</em>
</p>

---

## 📌 Countries Analyzed

- Norway 🇳🇴  
- Sweden 🇸🇪  
- Finland 🇫🇮  

---

## 🚀 Tools & Libraries

- Python
- pandas, numpy
- statsmodels (SARIMAX)
- xgboost
- shap
- matplotlib, seaborn
- scipy

---

## 👤 Author

**Samuel Kwesi Kumi**