# 🌍 Google Earth Engine (GEE) Scripts

This folder contains the Google Earth Engine (GEE) scripts used to extract and process weather data for the analysis.

## 📌 Purpose

The scripts in this folder are used to:

* Extract ERA5-Land temperature data
* Filter data for Nordic countries (Norway, Sweden, Finland)
* Generate daily weather variables used in demand modeling

These outputs form the basis of the dataset used in the forecasting models.

---

## 📂 Scripts

### `nordic_weather_era5.js`

* Loads ERA5-Land daily temperature data
* Filters observations for Nordic countries
* Aggregates temperature variables for analysis
* Exports data for further processing in Python

---

## ▶️ How to Run

1. Open the script in Google Earth Engine Code Editor

2. Update the date range if needed

3. Run the script

4. Export the data using:

   * `Export.table.toDrive()` or
   * `Export.image.toDrive()`

5. Save the exported file in the local `data/` folder:

```
data/nordic_energy_climate_df.csv
```

---

## 🔗 GEE Script Link

You can also access the script directly via GEE:
https://code.earthengine.google.com/eecce17b3b7bff34db99d6ec8df4d60a

---

## ⚠️ Notes

* Exported datasets are **not included** in this repository
* Users must generate the data using these scripts
* ERA5-Land data source: ECMWF / Copernicus Climate Data Store

---

## 📊 Role in Project Pipeline

```
GEE Scripts → Data Extraction → Local data/ folder → Python models
```

This ensures the project is fully reproducible from raw data extraction to final forecasting results.
