import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
# This section includes functions for creating log-transformed load features, lag features, temperature-based features
# and calendar dummies. The functions are designed to be flexible and can be easily applied to the dataset.
def create_log_features(df, countries):
    for country in countries.keys():
        df[f"log_load_{country}"] = np.log(df[f"Total Load [MW] - {country}"])
    return df


def create_lag_features(df, countries, max_lag=7):
    for country, code in countries.items():
        for lag in range(1, max_lag + 1):
            df[f"lag{lag}_log_{code}"] = df[f"log_load_{country}"].shift(lag)
    return df


def create_temperature_features(df):
    countries = ["Norway", "Sweden", "Finland"]

    for country in countries:
        df[f'HDD_{country}'] = np.maximum(0, 17 - df[f'Temp - {country}'])
        df[f'CDD_{country}'] = np.maximum(0, df[f'Temp - {country}'] - 21)

        cold_thresh = df[f'Temp - {country}'].quantile(0.05)
        warm_thresh = df[f'Temp - {country}'].quantile(0.95)

        code = country[:2].upper()

        df[f'Extreme_Cold_{code}'] = (df[f'Temp - {country}'] <= cold_thresh).astype(int)
        df[f'Extreme_Warm_{code}'] = (df[f'Temp - {country}'] >= warm_thresh).astype(int)

        df[f'HDD_Extreme_{code}'] = df[f'HDD_{country}'] * df[f'Extreme_Cold_{code}']
        df[f'CDD_Extreme_{code}'] = df[f'CDD_{country}'] * df[f'Extreme_Warm_{code}']

    return df

#Calendar dummies
# This function creates day-of-week and month dummies.

def create_calendar_dummies(df):
    weekday_order = [
        'Friday','Monday','Tuesday','Wednesday',
        'Thursday','Saturday','Sunday'
    ]

    df['Day of Week'] = pd.Categorical(df['Day of Week'], categories=weekday_order)

    day_dummies = pd.get_dummies(
        df['Day of Week'],
        prefix='day',
        drop_first=True,
        dtype=int
    )

    month_order = [
        'January','February','March','April','May','June',
        'July','August','September','October','November','December'
    ]

    df['Month'] = pd.Categorical(df['Month'], categories=month_order)

    month_dummies = pd.get_dummies(
        df['Month'],
        prefix='month',
        drop_first=True,
        dtype=int
    )

    df = pd.concat([df, day_dummies, month_dummies], axis=1)

    return df

#Train/validation split
# Note: This function is designed to be flexible for different training and forecasting years.

def split_data(df, train_start_year, forecast_year):
    val_year = forecast_year - 1
    train_end_year = forecast_year - 2

    train_data = df.loc[f'{train_start_year}-01-01':f'{train_end_year}-12-31'].copy()
    val_data   = df.loc[f'{val_year}-01-01':f'{val_year}-12-31'].copy()

    development_data = df.loc[f'{train_start_year}-01-01':f'{val_year}-12-31'].copy()
    forecast_data    = df.loc[f'{forecast_year}-01-01':f'{forecast_year}-12-31'].copy()

    return train_data, val_data, development_data, forecast_data