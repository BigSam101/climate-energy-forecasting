import pandas as pd
from statsmodels.tsa.stattools import adfuller


def run_adf_test(series):
    series = series.dropna()

    adf_stat, p_value, used_lag, n_obs, critical_values, icbest = adfuller(series)

    conclusion = "Stationary" if p_value < 0.05 else "Non-stationary"

    return {
        "ADF Statistic": adf_stat,
        "p-value": p_value,
        "Used lags": used_lag,
        "Observations": n_obs,
        "Conclusion": conclusion
    }


def run_adf_tests_by_country(
    data,
    countries=("Norway", "Sweden", "Finland"),
    transform="level",
    seasonal_lag=7
):
    results = {}

    for country in countries:
        col = f"log_load_{country}"
        series = data[col].copy()

        if transform == "first_difference":
            series = series.diff()
        elif transform == "seasonal_difference":
            series = series.diff(seasonal_lag)

        results[country] = run_adf_test(series)

    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "Country"})