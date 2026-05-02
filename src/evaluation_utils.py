import itertools
import os
import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_monthly_metrics(comparison_plot_df, country, forecast_year):
    rows = []

    for month in range(1, 13):
        monthly = comparison_plot_df[comparison_plot_df.index.month == month].copy()

        if monthly.empty:
            continue

        month_name = monthly.index[0].strftime("%B")

        for model_name in [col for col in monthly.columns if col != "Actual"]:
            rmse_log = np.sqrt(mean_squared_error(monthly["Actual"], monthly[model_name]))
            mae_log = mean_absolute_error(monthly["Actual"], monthly[model_name])

            rows.append({
                "Country": country,
                "Forecast Year": forecast_year,
                "Month": month_name,
                "Model": model_name,
                "RMSE (log)": rmse_log,
                "MAE (log)": mae_log
            })

    return pd.DataFrame(rows)


def calculate_annual_metrics(comparison_plot_df, country, forecast_year):
    rows = []

    for model_name in [col for col in comparison_plot_df.columns if col != "Actual"]:
        rmse_log = np.sqrt(mean_squared_error(comparison_plot_df["Actual"], comparison_plot_df[model_name]))
        mae_log = mean_absolute_error(comparison_plot_df["Actual"], comparison_plot_df[model_name])

        rows.append({
            "Country": country,
            "Forecast Year": forecast_year,
            "Model": model_name,
            "RMSE (log)": rmse_log,
            "MAE (log)": mae_log
        })

    annual_table = pd.DataFrame(rows)
    annual_table = annual_table.sort_values("RMSE (log)").reset_index(drop=True)
    annual_table["Rank"] = annual_table.index + 1

    return annual_table


def compute_basic_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


def run_volatility_analysis(
    comparison_plot_df,
    df,
    country,
    forecast_year,
    volatility_percentile_list=[90, 80, 75],
    temp_percentile_list=[90, 80, 75],
    pair_scenarios=True
):
    temp_col_map = {
        "Norway": "Temp - Norway",
        "Sweden": "Temp - Sweden",
        "Finland": "Temp - Finland"
    }

    if country not in temp_col_map:
        raise ValueError(f"Country '{country}' not found in temp_col_map.")

    temp_col = temp_col_map[country]

    model_order = [
        "SARIMAX Full",
        "SARIMAX Restricted",
        "Seasonal Naïve",
        "XGBoost No Weather",
        "XGBoost Full",
        "XGBoost Restricted"
    ]

    base_df = comparison_plot_df.copy().sort_index()
    base_df = base_df.join(
        df[[temp_col]].rename(columns={temp_col: "Temperature"}),
        how="left"
    )

    required_cols = ["Actual", "Temperature"] + [
        c for c in comparison_plot_df.columns if c != "Actual"
    ]

    base_df = base_df[required_cols].dropna().copy()

    base_df["abs_change_actual"] = base_df["Actual"].diff().abs()
    base_df["abs_change_temp"] = base_df["Temperature"].diff().abs()
    base_df = base_df.dropna().copy()

    model_cols = [col for col in comparison_plot_df.columns if col != "Actual"]

    scenarios = []

    if pair_scenarios:
        if len(volatility_percentile_list) != len(temp_percentile_list):
            raise ValueError("If pair_scenarios=True, both percentile lists must have the same length.")

        for vp, tp in zip(volatility_percentile_list, temp_percentile_list):
            scenarios.append({
                "volatility_percentile": vp,
                "temp_percentile": tp
            })
    else:
        for vp in volatility_percentile_list:
            for tp in temp_percentile_list:
                scenarios.append({
                    "volatility_percentile": vp,
                    "temp_percentile": tp
                })

    all_summary_tables = []
    all_performance_tables = []
    all_winner_tables = []
    plot_ready_tables = []

    for scen in scenarios:
        vp = scen["volatility_percentile"]
        tp = scen["temp_percentile"]

        scenario_df = base_df.copy()

        demand_cutoff = scenario_df["abs_change_actual"].quantile(vp / 100)
        scenario_df["volatile_day"] = scenario_df["abs_change_actual"] >= demand_cutoff

        temp_cutoff = scenario_df["abs_change_temp"].quantile(tp / 100)
        scenario_df["temp_shock"] = scenario_df["abs_change_temp"] >= temp_cutoff

        volatile_df = scenario_df[scenario_df["volatile_day"]].copy()

        if volatile_df.empty:
            continue

        volatile_df["Volatility Type"] = np.where(
            volatile_df["temp_shock"],
            "Temperature-Driven",
            "Other Factors"
        )

        summary_counts = (
            volatile_df["Volatility Type"]
            .value_counts()
            .rename_axis("Volatility Type")
            .reset_index(name="Days")
        )

        summary_counts["Country"] = country
        summary_counts["Forecast Year"] = forecast_year
        summary_counts["Volatility Percentile"] = vp
        summary_counts["Temperature Percentile"] = tp

        all_summary_tables.append(summary_counts)

        group_rows = []

        for group_name in ["Temperature-Driven", "Other Factors"]:
            group_df = volatile_df[volatile_df["Volatility Type"] == group_name].copy()

            if group_df.empty:
                continue

            for model in model_cols:
                metrics = compute_basic_metrics(group_df["Actual"], group_df[model])

                group_rows.append({
                    "Country": country,
                    "Forecast Year": forecast_year,
                    "Volatility Percentile": vp,
                    "Temperature Percentile": tp,
                    "Volatility Type": group_name,
                    "Days": len(group_df),
                    "Model": model,
                    **metrics
                })

        performance_df = pd.DataFrame(group_rows)

        if performance_df.empty:
            continue

        performance_df[["RMSE", "MAE"]] = performance_df[["RMSE", "MAE"]].round(3)
        performance_df = performance_df.sort_values(
            ["Volatility Type", "RMSE"]
        ).reset_index(drop=True)

        all_performance_tables.append(performance_df)

        winner_df = (
            performance_df
            .sort_values(["Volatility Type", "RMSE"])
            .groupby("Volatility Type", as_index=False)
            .first()[["Volatility Type", "Model", "Days", "RMSE", "MAE"]]
        )

        winner_df["Country"] = country
        winner_df["Forecast Year"] = forecast_year
        winner_df["Volatility Percentile"] = vp
        winner_df["Temperature Percentile"] = tp

        all_winner_tables.append(winner_df)

        plot_df = performance_df.copy()
        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=model_order,
            ordered=True
        )

        plot_df = plot_df.sort_values(["Model", "Volatility Type"]).reset_index(drop=True)

        plot_ready_tables.append({
            "country": country,
            "forecast_year": forecast_year,
            "volatility_percentile": vp,
            "temp_percentile": tp,
            "plot_df": plot_df
        })

    summary_counts_all_df = (
        pd.concat(all_summary_tables, ignore_index=True)
        if all_summary_tables else pd.DataFrame()
    )

    performance_by_type_all_df = (
        pd.concat(all_performance_tables, ignore_index=True)
        if all_performance_tables else pd.DataFrame()
    )

    winners_all_df = (
        pd.concat(all_winner_tables, ignore_index=True)
        if all_winner_tables else pd.DataFrame()
    )

    return summary_counts_all_df, performance_by_type_all_df, winners_all_df, plot_ready_tables


def significance_stars(p):
    if pd.isna(p):
        return ""
    elif p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def newey_west_long_run_variance(x, lag=None):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 2:
        return np.nan

    if lag is None:
        lag = int(np.floor(1.1447 * (n ** (1 / 3))))

    lag = min(lag, n - 1)

    x_centered = x - np.mean(x)
    gamma0 = np.dot(x_centered, x_centered) / n
    lrv = gamma0

    for h in range(1, lag + 1):
        gamma_h = np.dot(x_centered[h:], x_centered[:-h]) / n
        weight = 1 - h / (lag + 1)
        lrv += 2 * weight * gamma_h

    return lrv


def diebold_mariano_hac(y_true, y1, y2, power=2, lag=None, h=1):
    y_true = pd.Series(y_true).dropna()
    y1 = pd.Series(y1).dropna()
    y2 = pd.Series(y2).dropna()

    common_index = y_true.index.intersection(y1.index).intersection(y2.index)

    y_true = y_true.loc[common_index]
    y1 = y1.loc[common_index]
    y2 = y2.loc[common_index]

    e1 = y_true - y1
    e2 = y_true - y2

    d = (np.abs(e1) ** power) - (np.abs(e2) ** power)
    d = pd.Series(d).dropna()

    n = len(d)

    if n < 5:
        return np.nan, np.nan

    mean_d = d.mean()
    lrv = newey_west_long_run_variance(d.values, lag=lag)

    if pd.isna(lrv) or lrv <= 0:
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(lrv / n)

    h = int(h)
    if h < 1:
        h = 1

    hln_factor = np.sqrt((n + 1 - 2 * h + (h * (h - 1) / n)) / n)
    dm_stat_hln = dm_stat * hln_factor

    p_value = 2 * (1 - t.cdf(np.abs(dm_stat_hln), df=n - 1))

    return dm_stat_hln, p_value


def run_dm_tests(
    comparison_plot_df,
    country,
    forecast_year,
    models_to_compare=None,
    winter_months=[12, 1, 2, 3],
    lag=None,
    h=1,
    power=2
):
    if models_to_compare is None:
        models_to_compare = [
            "SARIMAX Full",
            "SARIMAX Restricted",
            "XGBoost Full",
            "XGBoost Restricted",
            "XGBoost No Weather"
        ]

    models_to_compare = [
        m for m in models_to_compare if m in comparison_plot_df.columns
    ]

    df_full = comparison_plot_df[["Actual"] + models_to_compare].dropna().copy()
    df_winter = df_full[df_full.index.month.isin(winter_months)].copy()

    dm_results = []

    def run_dm_table(df_in, period_name):
        for model_1, model_2 in itertools.combinations(models_to_compare, 2):
            dm_stat, p_val = diebold_mariano_hac(
                y_true=df_in["Actual"],
                y1=df_in[model_1],
                y2=df_in[model_2],
                power=power,
                lag=lag,
                h=h
            )

            if pd.isna(p_val):
                winner = "Not available"
            elif p_val < 0.05:
                winner = model_2 if dm_stat > 0 else model_1
            else:
                winner = "No significant difference"

            dm_results.append({
                "Country": country,
                "Forecast Year": forecast_year,
                "Period": period_name,
                "Model 1": model_1,
                "Model 2": model_2,
                "DM Stat": dm_stat,
                "p-value": p_val,
                "Significance": significance_stars(p_val),
                "Winner (5%)": winner
            })

    run_dm_table(df_full, "Full Year")
    run_dm_table(df_winter, "Winter")

    dm_table = pd.DataFrame(dm_results)

    dm_table["DM Stat"] = dm_table["DM Stat"].round(3)
    dm_table["p-value"] = dm_table["p-value"].round(4)

    dm_table["p-value (stars)"] = dm_table.apply(
        lambda row: f"{row['p-value']:.4f}{row['Significance']}"
        if pd.notna(row["p-value"]) else "",
        axis=1
    )

    dm_table = dm_table.sort_values(
        by=["Period", "p-value"]
    ).reset_index(drop=True)

    dm_table_clean = dm_table[
        ["Country", "Forecast Year", "Period", "Model 1", "Model 2",
         "DM Stat", "p-value (stars)", "Winner (5%)"]
    ].copy()

    return dm_table_clean


def save_evaluation_outputs(
    country,
    forecast_year,
    annual_table,
    monthly_table,
    summary_counts_all_df,
    performance_by_type_all_df,
    winners_all_df,
    dm_table_clean,
    output_dir="outputs/tables"
):
    os.makedirs(output_dir, exist_ok=True)

    safe_country = str(country).replace(" ", "_")

    excel_path = os.path.join(
        output_dir,
        f"{safe_country}_model_evaluation_{forecast_year}.xlsx"
    )

    with pd.ExcelWriter(excel_path) as writer:
        annual_table.to_excel(writer, sheet_name="Annual Metrics", index=False)
        monthly_table.to_excel(writer, sheet_name="Monthly Metrics", index=False)
        summary_counts_all_df.to_excel(writer, sheet_name="Volatility Counts", index=False)
        performance_by_type_all_df.to_excel(writer, sheet_name="Volatility Performance", index=False)
        winners_all_df.to_excel(writer, sheet_name="Volatility Winners", index=False)
        dm_table_clean.to_excel(writer, sheet_name="DM Tests", index=False)

    return excel_path