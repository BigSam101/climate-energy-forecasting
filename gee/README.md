# 🌍 Google Earth Engine (GEE) Scripts

This folder contains GEE scripts used to extract and process weather data for the analysis.

---

## 📌 Purpose

The scripts are used to:

* Extract ERA5-Land daily temperature data
* Compute **population-weighted temperature** using WorldPop data
* Aggregate temperature at the country level (Norway, Sweden, Finland)
* Export daily time series for use in forecasting models

This ensures that temperature reflects **where people actually live**, rather than simple geographic averages.

---

## 📂 Scripts

### `nordic_weather_era5.js`

This script performs the following steps:

1. Defines the study period (2010–2025)
2. Loads Nordic country boundaries
3. Extracts ERA5-Land daily temperature (2m)
4. Loads **WorldPop population raster (100m resolution)**
5. Computes **population-weighted temperature**:

   * Temperature × population weights
   * Aggregated using weighted averages
6. Builds a daily panel dataset by country
7. Exports results

---

## 📊 Data Sources

* **ERA5-Land** (ECMWF / Copernicus)
  → Daily temperature data

* **WorldPop**
  → High-resolution population distribution used for weighting

---

## ▶️ How to Run

1. Open the script in the GEE Code Editor
2. Run the script
3. Export data via:

```id="g1"
Export.table.toDrive(...)
```

4. Download the exported file and place it in:

```id="g2"
data/nordic_energy_climate_df.csv
```

---

## 🔗 GEE Script

(https://code.earthengine.google.com/eecce17b3b7bff34db99d6ec8df4d60a)

---

## ⚠️ Notes

* Exported datasets are **not included** in this repository
* Users must generate the dataset using the scripts
* Population-weighted temperature improves demand modeling accuracy

---

## 🔁 Pipeline

```id="g3"
GEE (ERA5 + WorldPop)
        ↓
Population-weighted temperature
        ↓
data/ folder (local)
        ↓
Python models (SARIMAX, XGBoost)
```
