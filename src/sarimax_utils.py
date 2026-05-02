import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from metrics import rmse, mae, compute_metrics


def fit_sarimax_model(
    train_data,
    target_col,
    exog_vars=None,
    order=(0, 1, 1),
    seasonal_order=(1, 1, 1, 7)
):
    cols = [target_col] + (exog_vars if exog_vars is not None else [])
    df_model = train_data[cols].dropna().copy()

    y = df_model[target_col]
    X = df_model[exog_vars] if exog_vars is not None else None

    model = SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False, cov_type="robust")
    return results


def forecast_sarimax_model(results, test_data, target_col, exog_vars=None):
    cols = [target_col] + (exog_vars if exog_vars is not None else [])
    df_test = test_data[cols].dropna().copy()

    y_test_log = df_test[target_col]
    X_test = df_test[exog_vars] if exog_vars is not None else None

    forecast = results.get_forecast(
        steps=len(y_test_log),
        exog=X_test
    )

    y_pred_log = forecast.predicted_mean
    y_test_log, y_pred_log = y_test_log.align(y_pred_log, join="inner")

    return y_test_log, y_pred_log


def sarimax_grid_search(
    train_data,
    val_data,
    development_data,
    specs,
    country,
    target_col,
    forecast_year,
    val_year,
    train_start_year,
    train_end_year,
    p=range(0, 2),
    q=range(0, 2),
    P=range(0, 2),
    Q=range(0, 2),
    d=0,
    D=1,
    s=7,
    top_n=5
):
    results_list = []
    selected_models = {}
    final_results_dict = {}

    for spec_name, exog_vars in specs.items():

        train_model = train_data[[target_col] + exog_vars].dropna().copy()
        val_model = val_data[[target_col] + exog_vars].dropna().copy()

        y_train = train_model[target_col]
        X_train = train_model[exog_vars]

        y_val = val_model[target_col]
        X_val = val_model[exog_vars]

        spec_results = []

        for order in itertools.product(p, [d], q):
            for seasonal_order in itertools.product(P, [D], Q, [s]):

                try:
                    model = SARIMAX(
                        y_train,
                        exog=X_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    results = model.fit(disp=False, cov_type="robust")

                    forecast = results.get_forecast(
                        steps=len(y_val),
                        exog=X_val
                    )

                    y_pred_log = forecast.predicted_mean
                    y_val_aligned, y_pred_log = y_val.align(y_pred_log, join="inner")

                    rmse_val = rmse(y_val_aligned, y_pred_log)
                    mae_val = mae(y_val_aligned, y_pred_log)

                    row = {
                        "Country": country,
                        "Specification": spec_name,
                        "Forecast Year": forecast_year,
                        "Validation Year": val_year,
                        "Train Start Year": train_start_year,
                        "Train End Year": train_end_year,
                        "Target": target_col,
                        "Exogenous Variables": ", ".join(exog_vars),
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "AIC": results.aic,
                        "BIC": results.bic,
                        "RMSE (log)": rmse_val,
                        "MAE (log)": mae_val
                    }

                    spec_results.append(row)
                    results_list.append(row)

                except Exception:
                    continue

        spec_grid_results = pd.DataFrame(spec_results)

        if spec_grid_results.empty:
            continue

        spec_grid_results = spec_grid_results.sort_values(
            by=["RMSE (log)", "MAE (log)", "AIC", "BIC"]
        ).reset_index(drop=True)

        spec_grid_results["RMSE Rank"] = spec_grid_results["RMSE (log)"].rank(method="dense").astype(int)
        spec_grid_results["MAE Rank"] = spec_grid_results["MAE (log)"].rank(method="dense").astype(int)
        spec_grid_results["AIC Rank"] = spec_grid_results["AIC"].rank(method="dense").astype(int)
        spec_grid_results["BIC Rank"] = spec_grid_results["BIC"].rank(method="dense").astype(int)

        spec_grid_results["Combined Score"] = (
            spec_grid_results["RMSE Rank"] +
            spec_grid_results["MAE Rank"] +
            spec_grid_results["AIC Rank"] +
            spec_grid_results["BIC Rank"]
        )

        top_models = spec_grid_results.nsmallest(top_n, "RMSE (log)").copy()
        selected_idx = top_models["AIC"].idxmin()
        selected_model = top_models.loc[selected_idx]

        selected_models[spec_name] = selected_model

        best_order = selected_model["order"]
        best_seasonal_order = selected_model["seasonal_order"]

        final_results = fit_sarimax_model(
            train_data=development_data,
            target_col=target_col,
            exog_vars=exog_vars,
            order=best_order,
            seasonal_order=best_seasonal_order
        )

        final_results_dict[spec_name] = final_results

    grid_results = pd.DataFrame(results_list)

    selected_summary_df = pd.DataFrame([
        {
            "Country": country,
            "Specification": spec_name,
            "Forecast Year": forecast_year,
            "Validation Year": val_year,
            "order": selected_model["order"],
            "seasonal_order": selected_model["seasonal_order"],
            "RMSE (log)": selected_model["RMSE (log)"],
            "MAE (log)": selected_model["MAE (log)"],
            "AIC": selected_model["AIC"],
            "BIC": selected_model["BIC"]
        }
        for spec_name, selected_model in selected_models.items()
    ])

    return grid_results, selected_summary_df, selected_models, final_results_dict


def compare_sarimax_models(
    model_specs,
    development_data,
    forecast_data,
    target_col,
    country
):
    fitted_models = {}
    comparison_rows = []

    for model_name, spec in model_specs.items():
        results = fit_sarimax_model(
            train_data=development_data,
            target_col=target_col,
            exog_vars=spec["exog_vars"],
            order=spec["order"],
            seasonal_order=spec["seasonal_order"]
        )

        fitted_models[model_name] = results

        y_true, y_pred = forecast_sarimax_model(
            results=results,
            test_data=forecast_data,
            target_col=target_col,
            exog_vars=spec["exog_vars"]
        )

        metrics = compute_metrics(y_true, y_pred)

        comparison_rows.append({
            "Country": country,
            "Model": model_name,
            "Specification": spec["Specification"],
            "Order": str(spec["order"]),
            "Seasonal Order": str(spec["seasonal_order"]),
            **metrics
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("RMSE").reset_index(drop=True)
    comparison_df["Rank"] = comparison_df.index + 1

    return comparison_df, fitted_models