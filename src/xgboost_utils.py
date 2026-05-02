import pandas as pd
import os
import numpy as np
import shap
from xgboost import XGBRegressor


def get_xgboost_feature_sets(
    df,
    development_data,
    country,
    lag_prefix,
    full_exog_vars,
    restricted_exog_vars
):
    lag_cols = [f"lag{i}_log_{lag_prefix}" for i in range(1, 8)]
    day_dummy_cols = [col for col in df.columns if col.startswith("day_")]
    month_dummy_cols = [col for col in df.columns if col.startswith("month_")]
    holiday_col = f"Holiday_{country}"

    xgb_full_features = full_exog_vars + day_dummy_cols + month_dummy_cols + lag_cols
    xgb_restricted_features = restricted_exog_vars + day_dummy_cols + month_dummy_cols + lag_cols
    xgb_no_weather_features = [holiday_col] + day_dummy_cols + month_dummy_cols + lag_cols

    def clean_features(features):
        features = [col for col in features if col in development_data.columns]
        return list(dict.fromkeys(features))

    return {
        "XGBoost Full": clean_features(xgb_full_features),
        "XGBoost Restricted": clean_features(xgb_restricted_features),
        "XGBoost No Weather": clean_features(xgb_no_weather_features)
    }


def prepare_xgboost_data(development_data, forecast_data, target_col, features):
    train = development_data[[target_col] + features].dropna().copy()
    test = forecast_data[[target_col] + features].dropna().copy()

    X_train = train[features]
    y_train = train[target_col]

    X_test = test[features]
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test


def fit_xgboost_model(X_train, y_train, random_state=42):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model


def fit_predict_xgboost_models(
    development_data,
    forecast_data,
    target_col,
    feature_sets,
    random_state=42
):
    fitted_models = {}
    predictions = {}
    test_targets = {}

    for model_name, features in feature_sets.items():
        X_train, y_train, X_test, y_test = prepare_xgboost_data(
            development_data=development_data,
            forecast_data=forecast_data,
            target_col=target_col,
            features=features
        )

        model = fit_xgboost_model(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state
        )

        pred = pd.Series(
            model.predict(X_test),
            index=X_test.index,
            name=model_name
        )

        fitted_models[model_name] = model
        predictions[model_name] = pred
        test_targets[model_name] = y_test

    return predictions, fitted_models, test_targets


def seasonal_naive_forecast(df, target_col, forecast_data, seasonal_lag=7):
    pred = df[target_col].shift(seasonal_lag).loc[forecast_data.index]
    pred.name = "Seasonal Naïve"
    return pred


def build_sarimax_forecasts(forecast_data, target_col, model_specs, fitted_sarimax_models):
    predictions = {}
    y_test_log = forecast_data[target_col].copy()

    for model_name, spec in model_specs.items():
        exog_vars = spec["exog_vars"]
        results = fitted_sarimax_models[model_name]

        X_test = forecast_data.loc[y_test_log.index, exog_vars]

        pred = results.get_forecast(
            steps=len(y_test_log),
            exog=X_test
        ).predicted_mean

        pred.name = model_name
        predictions[model_name] = pred

    return predictions


def build_comparison_plot_df(y_test_log, predictions_log):
    comparison_plot_df = pd.DataFrame({"Actual": y_test_log})

    for model_name, pred in predictions_log.items():
        comparison_plot_df[model_name] = pred.reindex(y_test_log.index)

    comparison_plot_df = comparison_plot_df.dropna()

    return comparison_plot_df


# SHAP FEATURE IMPORTANCE 



def get_xgb_gain_importance(model, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)


def get_xgb_shap_importance(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    return shap_values, shap_importance


def make_pretty_feature_names(country, code, lag_prefix):
    return {
        **{f"lag{i}_log_{lag_prefix}": f"Lag {i}" for i in range(1, 8)},
        f"HDD_{country}": "HDD",
        f"Extreme_Cold_{code}": "Extreme Cold",
        f"HDD_Extreme_{code}": "HDD × Extreme Cold",
        f"CDD_{country}": "CDD",
        f"Extreme_Warm_{code}": "Extreme Warm",
        f"CDD_Extreme_{code}": "CDD × Extreme Warm",
        f"Holiday_{country}": "Holiday",
        "day_Monday": "Monday",
        "day_Tuesday": "Tuesday",
        "day_Wednesday": "Wednesday",
        "day_Thursday": "Thursday",
        "day_Saturday": "Saturday",
        "day_Sunday": "Sunday",
        "month_February": "February",
        "month_March": "March",
        "month_April": "April",
        "month_May": "May",
        "month_June": "June",
        "month_July": "July",
        "month_August": "August",
        "month_September": "September",
        "month_October": "October",
        "month_November": "November",
        "month_December": "December",
    }


def build_shap_comparison_table(shap_importance_dict, pretty_feature_names):
    renamed_tables = []

    rename_map = {
        "XGBoost Full": "MeanAbsSHAP_Full",
        "XGBoost Restricted": "MeanAbsSHAP_Restricted",
        "XGBoost No Weather": "MeanAbsSHAP_NoWeather"
    }

    for model_name, shap_df in shap_importance_dict.items():
        temp = shap_df.rename(columns={"MeanAbsSHAP": rename_map[model_name]})
        renamed_tables.append(temp)

    shap_compare = renamed_tables[0]

    for table in renamed_tables[1:]:
        shap_compare = pd.merge(shap_compare, table, on="Feature", how="outer")

    shap_compare = shap_compare.fillna(0)

    shap_compare["PrettyFeature"] = (
        shap_compare["Feature"]
        .map(pretty_feature_names)
        .fillna(shap_compare["Feature"])
    )

    shap_cols = [
        "MeanAbsSHAP_Full",
        "MeanAbsSHAP_Restricted",
        "MeanAbsSHAP_NoWeather"
    ]

    shap_compare["TotalImportance"] = shap_compare[shap_cols].sum(axis=1)

    return shap_compare.sort_values("TotalImportance", ascending=False).reset_index(drop=True)


def save_shap_outputs(
    country,
    forecast_year,
    gain_importance_dict,
    shap_importance_dict,
    shap_compare,
    output_dir="outputs/shap"
):
    os.makedirs(output_dir, exist_ok=True)

    safe_country = str(country).replace(" ", "_")

    excel_path = os.path.join(
        output_dir,
        f"{safe_country}_xgboost_shap_importance_{forecast_year}.xlsx"
    )

    csv_path = os.path.join(
        output_dir,
        f"shap_compare_{safe_country}_{forecast_year}.csv"
    )

    with pd.ExcelWriter(excel_path) as writer:
        for model_name, df in gain_importance_dict.items():
            sheet_name = model_name.replace("XGBoost ", "Gain ")[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for model_name, df in shap_importance_dict.items():
            sheet_name = model_name.replace("XGBoost ", "SHAP ")[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    shap_compare.to_csv(csv_path, index=False)

    return excel_path, csv_path