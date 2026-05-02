import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_dir(output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)


def plot_actual_vs_forecasts(
    comparison_plot_df,
    country,
    forecast_year,
    models=None,
    output_dir="outputs/figures",
    save=True
):
    ensure_output_dir(output_dir)

    if models is None:
        models = [col for col in comparison_plot_df.columns if col != "Actual"]

    plt.figure(figsize=(14, 6))

    plt.plot(
        comparison_plot_df.index,
        comparison_plot_df["Actual"],
        label="Actual",
        linewidth=2
    )

    for model in models:
        if model in comparison_plot_df.columns:
            plt.plot(
                comparison_plot_df.index,
                comparison_plot_df[model],
                label=model,
                linewidth=1.5,
                alpha=0.85
            )

    plt.title(f"Actual vs Forecasted Electricity Demand - {country} ({forecast_year})", fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Log electricity demand")
    plt.legend(frameon=False, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        file_path = os.path.join(output_dir, f"{country}_actual_vs_forecast_{forecast_year}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()


def plot_monthly_rmse(
    monthly_table,
    country,
    forecast_year,
    output_dir="outputs/figures",
    save=True
):
    ensure_output_dir(output_dir)

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    plot_df = monthly_table.copy()
    plot_df["Month"] = pd.Categorical(plot_df["Month"], categories=month_order, ordered=True)
    plot_df = plot_df.sort_values("Month")

    pivot_df = plot_df.pivot(index="Month", columns="Model", values="RMSE (log)")

    ax = pivot_df.plot(kind="bar", figsize=(15, 6), rot=45)

    ax.set_title(f"Monthly RMSE by Model - {country} ({forecast_year})", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE (log)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        file_path = os.path.join(output_dir, f"{country}_monthly_rmse_{forecast_year}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()


def plot_volatility_scenarios(
    plot_ready_tables,
    country,
    forecast_year,
    output_dir="outputs/figures",
    save=True
):
    ensure_output_dir(output_dir)

    model_label_map = {
        "SARIMAX Full": "SARIMAX\nFull",
        "SARIMAX Restricted": "SARIMAX\nRestricted",
        "Seasonal Naïve": "Seasonal\nNaïve",
        "XGBoost No Weather": "XGBoost\nNo Weather",
        "XGBoost Full": "XGBoost\nFull",
        "XGBoost Restricted": "XGBoost\nRestricted"
    }

    model_order = [
        "SARIMAX Full",
        "SARIMAX Restricted",
        "Seasonal Naïve",
        "XGBoost No Weather",
        "XGBoost Full",
        "XGBoost Restricted"
    ]

    color_map = {
        "Temperature-Driven": "#1f21b4",
        "Other Factors": "#ff7f0e"
    }

    n = len(plot_ready_tables)

    if n == 0:
        raise ValueError("No valid volatility scenarios to plot.")

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)

    if n == 1:
        axes = [axes]

    for ax, scen_plot in zip(axes, plot_ready_tables):
        vp = scen_plot["volatility_percentile"]
        tp = scen_plot["temp_percentile"]
        plot_df = scen_plot["plot_df"].copy()

        plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_order, ordered=True)
        plot_df = plot_df.sort_values(["Model", "Volatility Type"]).reset_index(drop=True)
        plot_df["Model_Label"] = plot_df["Model"].astype(str).map(model_label_map)

        group_counts = (
            plot_df[["Volatility Type", "Days"]]
            .drop_duplicates()
            .set_index("Volatility Type")["Days"]
            .to_dict()
        )

        pivot_plot = plot_df.pivot(
            index="Model_Label",
            columns="Volatility Type",
            values="RMSE"
        )

        desired_label_order = [
            model_label_map[m] for m in model_order
            if m in plot_df["Model"].astype(str).unique()
        ]

        pivot_plot = pivot_plot.reindex(desired_label_order)

        for needed_col in ["Temperature-Driven", "Other Factors"]:
            if needed_col not in pivot_plot.columns:
                pivot_plot[needed_col] = np.nan

        pivot_plot = pivot_plot[["Temperature-Driven", "Other Factors"]]

        pivot_plot.columns = [
            f"Temperature-Driven ({group_counts.get('Temperature-Driven', 0)} days)",
            f"Other Factors ({group_counts.get('Other Factors', 0)} days)"
        ]

        pivot_plot.plot(
            kind="bar",
            ax=ax,
            rot=0,
            color=[color_map["Temperature-Driven"], color_map["Other Factors"]]
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

        ax.set_title(
            f"Demand Volatility Q≥{vp/100:.2f}\nTemperature Shocks Q≥{tp/100:.2f}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("RMSE" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="", fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Forecast Performance by Volatility Type - {country} ({forecast_year})",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    if save:
        file_path = os.path.join(output_dir, f"{country}_volatility_scenarios_{forecast_year}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()


def plot_shap_country_comparison(
    input_dir,
    forecast_year,
    countries=("Norway", "Sweden", "Finland"),
    top_n=10,
    output_dir="outputs/figures",
    save=True
):
    ensure_output_dir(output_dir)

    colors = {
        "full": "#911fb4",
        "restricted": "#0e6eff",
        "no_weather": "#a0812c"
    }

    fig, axes = plt.subplots(1, len(countries), figsize=(24, 8), sharex=False)

    if len(countries) == 1:
        axes = [axes]

    for ax, country_name in zip(axes, countries):
        safe_country = country_name.replace(" ", "_")
        file_path = os.path.join(input_dir, f"shap_compare_{safe_country}_{forecast_year}.csv")

        shap_df = pd.read_csv(file_path)

        shap_df_top = (
            shap_df.sort_values("TotalImportance", ascending=False)
            .head(top_n)
            .sort_values("TotalImportance", ascending=True)
        )

        y = np.arange(len(shap_df_top))
        bar_h = 0.25

        ax.barh(
            y - bar_h,
            shap_df_top["MeanAbsSHAP_Full"],
            height=bar_h,
            label="XGBoost Full",
            color=colors["full"]
        )

        ax.barh(
            y,
            shap_df_top["MeanAbsSHAP_Restricted"],
            height=bar_h,
            label="XGBoost Restricted",
            color=colors["restricted"]
        )

        ax.barh(
            y + bar_h,
            shap_df_top["MeanAbsSHAP_NoWeather"],
            height=bar_h,
            label="XGBoost No Weather",
            color=colors["no_weather"]
        )

        ax.set_yticks(y)
        ax.set_yticklabels(shap_df_top["PrettyFeature"], fontweight="bold")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(country_name, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        f"Feature Importance Comparison Across Countries ({forecast_year})",
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save:
        file_path = os.path.join(output_dir, f"shap_country_comparison_{forecast_year}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()
    
    
# Additional utility functions for descriptive figures and tables
    
def plot_load_distributions(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    countries = ["Norway", "Sweden", "Finland"]

    load_cols = {
        "Norway": "Total Load [MW] - Norway",
        "Sweden": "Total Load [MW] - Sweden",
        "Finland": "Total Load [MW] - Finland"
    }

    colors = {
        "Norway": "#ff4d4d",
        "Sweden": "#4dc3ff",
        "Finland": "#4daf4a"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, country in enumerate(countries):
        sns.histplot(
            df[load_cols[country]].dropna(),
            bins=30,
            color=colors[country],
            ax=axes[i]
        )

        axes[i].set_title(f"{country}")
        axes[i].grid(alpha=0.3)

    plt.suptitle("Electricity Load Distribution", fontweight="bold")
    plt.tight_layout()
    plt.show()
    
# Temperature vs Load
# This is a simple scatter plot with a regression line to visualize the relationship between temperature and electricity load for each country.
# It can help illustrate the temperature sensitivity of demand, which is a key motivation for our modeling approach. 
    
def plot_temp_vs_load(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    countries = ["Norway", "Sweden", "Finland"]

    for i, country in enumerate(countries):
        sns.regplot(
            x=f"Temp - {country}",
            y=f"Total Load [MW] - {country}",
            data=df,
            ax=axes[i],
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"}
        )

        axes[i].set_title(country)
        axes[i].grid(alpha=0.3)

    plt.suptitle("Temperature vs Electricity Load", fontweight="bold")
    plt.tight_layout()
    plt.show()
    
# HDD / CDD relationships
def plot_hdd_cdd_relationships(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    countries = ["Norway", "Sweden", "Finland"]

    hdd_map = {
        "Norway": "HDD_Norway",
        "Sweden": "HDD_Sweden",
        "Finland": "HDD_Finland"
    }

    cdd_map = {
        "Norway": "CDD_Norway",
        "Sweden": "CDD_Sweden",
        "Finland": "CDD_Finland"
    }

    load_map = {
        "Norway": "Total Load [MW] - Norway",
        "Sweden": "Total Load [MW] - Sweden",
        "Finland": "Total Load [MW] - Finland"
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    for i, country in enumerate(countries):

        # HDD
        sns.regplot(
            x=df[hdd_map[country]],
            y=df[load_map[country]],
            ax=axes[0, i],
            scatter_kws={'alpha': 0.2},
            line_kws={'color': 'black'}
        )

        axes[0, i].set_title(country)
        axes[0, i].set_xlabel("HDD")

        # CDD
        sns.regplot(
            x=df[cdd_map[country]],
            y=df[load_map[country]],
            ax=axes[1, i],
            scatter_kws={'alpha': 0.2},
            line_kws={'color': 'black'}
        )

        axes[1, i].set_xlabel("CDD")

    axes[0, 0].set_ylabel("Load")
    axes[1, 0].set_ylabel("Load")

    plt.suptitle("HDD and CDD vs Load", fontweight="bold")
    plt.tight_layout()
    plt.show()
    
# Extreme temperature boxplots

def plot_extreme_boxplots(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    countries = ["Norway", "Sweden", "Finland"]

    extreme_cold = {
        "Norway": "Extreme_Cold_NO",
        "Sweden": "Extreme_Cold_SE",
        "Finland": "Extreme_Cold_FI"
    }

    load_map = {
        "Norway": "Total Load [MW] - Norway",
        "Sweden": "Total Load [MW] - Sweden",
        "Finland": "Total Load [MW] - Finland"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, country in enumerate(countries):
        sns.boxplot(
            x=df[extreme_cold[country]],
            y=df[load_map[country]],
            ax=axes[i]
        )

        axes[i].set_title(country)

    plt.suptitle("Extreme Cold vs Load", fontweight="bold")
    plt.tight_layout()
    plt.show()
    
# ADF test results by country and transformation type

def plot_acf_pacf_by_country(
    data,
    countries=("Norway", "Sweden", "Finland"),
    transform="level",
    seasonal_lag=7,
    lags=60,
    title=None
):
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from stat_tests import run_adf_test

    country_colors = {
        "Norway": "#e41a1c",
        "Sweden": "#377eb8",
        "Finland": "#4daf4a"
    }

    transform_titles = {
        "level": "Daily Log Load",
        "first_difference": "First Differenced Daily Log Load",
        "seasonal_difference": f"Seasonal Difference ({seasonal_lag} Days)"
    }

    if title is None:
        title = f"ACF and PACF - {transform_titles[transform]}"

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), facecolor="#f5f7fb")

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        color="black"
    )

    adf_results = {}

    for i, country in enumerate(countries):
        col = f"log_load_{country}"
        series = data[col].copy()

        if transform == "first_difference":
            series = series.diff()
            subtitle_suffix = " (1st Difference)"
        elif transform == "seasonal_difference":
            series = series.diff(seasonal_lag)
            subtitle_suffix = ""
        else:
            subtitle_suffix = ""

        series = series.dropna()
        color = country_colors[country]

        adf_result = run_adf_test(series)
        adf_results[country] = adf_result

        plot_acf(
            series,
            ax=axes[i, 0],
            lags=lags,
            alpha=0.05,
            zero=False,
            vlines_kwargs={"colors": color, "linewidth": 2}
        )

        axes[i, 0].set_title(
            f"ACF - {country}{subtitle_suffix}",
            fontsize=14,
            fontweight="bold",
            color="black"
        )

        plot_pacf(
            series,
            ax=axes[i, 1],
            lags=lags,
            method="ywm",
            alpha=0.05,
            zero=False,
            vlines_kwargs={"colors": color, "linewidth": 2}
        )

        axes[i, 1].set_title(
            f"PACF - {country}{subtitle_suffix}",
            fontsize=14,
            fontweight="bold",
            color="black"
        )

        adf_text = (
            f"ADF Stat: {adf_result['ADF Statistic']:.3f}\n"
            f"p-value: {adf_result['p-value']:.4g}\n"
            f"Lags: {adf_result['Used lags']}\n"
            f"Obs: {adf_result['Observations']}\n"
            f"{adf_result['Conclusion']}"
        )

        axes[i, 1].text(
            1.05,
            0.5,
            adf_text,
            transform=axes[i, 1].transAxes,
            fontsize=10,
            color="black",
            verticalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f8f8f8",
                edgecolor=color,
                linewidth=1.5
            )
        )

        for ax in [axes[i, 0], axes[i, 1]]:
            ax.set_facecolor("#ffffff")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", colors="black", labelsize=10)

            for line in ax.lines:
                if len(line.get_xdata()) > 2:
                    line.set_color("black")
                    line.set_linewidth(1.2)
                else:
                    line.set_color(color)
                    line.set_markerfacecolor(color)
                    line.set_markeredgecolor(color)
                    line.set_markersize(5)

            for coll in ax.collections:
                try:
                    coll.set_facecolor(color)
                    coll.set_edgecolor(color)
                    coll.set_alpha(0.18)
                except Exception:
                    pass

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    plt.show()

    return pd.DataFrame(adf_results).T.reset_index().rename(columns={"index": "Country"})

# SARIMAX variable effect distributions
# This function simulates the coefficient distributions for specified SARIMAX variables based on their estimated coefficients and standard errors. 
# It then plots the density of these simulated effects for each country, allowing for a visual comparison of the baseline and extreme cold demand sensitivities across countries. 
# The resulting plot is saved to the specified output directory.
def plot_sarimax_variable_effect_distributions(
    files,
    effects_to_plot=None,
    n_sim=10000,
    seed=42,
    output_dir="outputs/figures",
    save=True
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    os.makedirs(output_dir, exist_ok=True)

    if effects_to_plot is None:
        effects_to_plot = {
            "HDD": "Baseline HDD effect",
            "Extreme Cold": "Extreme cold effect"
        }

    country_colors = {
        "Norway": "#2166ac",
        "Sweden": "#1b9e77",
        "Finland": "#d95f02"
    }

    np.random.seed(seed)

    sim_rows = []

    for country, path in files.items():
        res = pd.read_csv(path)

        for var, label in effects_to_plot.items():
            row = res[res["Pretty Variable"] == var].iloc[0]

            draws = np.random.normal(
                row["Coefficient"],
                row["Std_Error"],
                n_sim
            )

            sim_rows.append(pd.DataFrame({
                "Country": country,
                "Effect": label,
                "Simulated effect": draws
            }))

    sim_df = pd.concat(sim_rows, ignore_index=True)

    fig, axes = plt.subplots(1, len(effects_to_plot), figsize=(14, 5), sharey=False)

    if len(effects_to_plot) == 1:
        axes = [axes]

    for ax, (var, label) in zip(axes, effects_to_plot.items()):

        for country in files.keys():
            vals = sim_df.loc[
                (sim_df["Country"] == country) &
                (sim_df["Effect"] == label),
                "Simulated effect"
            ]

            density = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 600)
            ys = density(xs)

            ax.plot(
                xs,
                ys,
                linewidth=2.7,
                color=country_colors.get(country),
                label=country
            )

            ax.fill_between(
                xs,
                ys,
                color=country_colors.get(country),
                alpha=0.18
            )

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Effect on log electricity demand")
        ax.set_ylabel("Density")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Country",
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle(
        "Baseline and Extreme Cold Demand Sensitivity by Country",
        fontsize=15,
        fontweight="bold",
        y=1.08
    )

    plt.tight_layout()

    if save:
        file_path = os.path.join(
            output_dir,
            "sarimax_variable_effect_distributions.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()

    return sim_df


# Monthly forecast panels
# This function creates a grid of monthly forecast panels for a specified country and forecast year. Each panel compares the actual electricity load against various forecast models for a given month. 
# The function allows for customization of the months to include, the scale of the y-axis (log or GWh), and the duration of months to focus on (e.g., winter peak, summer trough).

def plot_monthly_forecast_panels(
    comparison_plot_df,
    country,
    forecast_year,
    plot_scale="log",
    duration="winter_peak",
    custom_months=None,
    output_dir="outputs/figures",
    save=True
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from sklearn.metrics import mean_squared_error

    os.makedirs(output_dir, exist_ok=True)

    if custom_months is None:
        custom_months = [1, 2, 3, 11, 12]

    duration_map = {
        "all": list(range(1, 13)),
        "winter_peak": [1, 2, 3, 10, 11, 12],
        "summer_trough": [5, 6, 7, 8, 9],
        "custom": custom_months
    }

    duration_title_map = {
        "all": "Monthly Forecast Comparison",
        "winter_peak": "Peak Monthly Forecasts",
        "summer_trough": "Summer Trough Forecast Comparison",
        "custom": "Selected Months Forecast Comparison"
    }

    if duration.lower() not in duration_map:
        raise ValueError("duration must be one of: 'all', 'winter_peak', 'summer_trough', 'custom'")

    selected_months = duration_map[duration.lower()]
    plot_title = duration_title_map[duration.lower()]

    plot_df = comparison_plot_df.copy()

    if plot_scale.lower() == "gwh":
        plot_df = np.exp(plot_df) / 1000
        y_label = "Electricity Load (GWh)"
    elif plot_scale.lower() == "log":
        y_label = "Log Electricity Load"
    else:
        raise ValueError("plot_scale must be either 'log' or 'gwh'")

    colors = {
        "Actual": "black",
        "SARIMAX Full": "green",
        "SARIMAX Restricted": "red",
        "XGBoost Full": "purple",
        "XGBoost Restricted": "blue",
        "XGBoost No Weather": "brown",
        "Seasonal Naïve": "orange"
    }

    def rmse(y, yhat):
        return np.sqrt(mean_squared_error(y, yhat))

    if duration.lower() == "all":
        nrows, ncols = 4, 3
        figsize = (22, 15)
    else:
        nrows, ncols = 3, 2
        figsize = (18, 16)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    axes = np.array(axes).flatten()

    for i, month in enumerate(selected_months):
        ax = axes[i]
        monthly = plot_df[plot_df.index.month == month]

        if monthly.empty:
            ax.set_visible(False)
            continue

        x_vals = monthly.index.day

        for col in plot_df.columns:
            ax.plot(
                x_vals,
                monthly[col],
                label=col,
                color=colors.get(col, "gray"),
                linestyle="-" if col == "Actual" else "--",
                linewidth=2 if col == "Actual" else 1.6,
                alpha=1.0 if col == "Actual" else 0.9
            )

        metrics = {
            col: rmse(monthly["Actual"], monthly[col])
            for col in plot_df.columns if col != "Actual"
        }

        best_model = min(metrics, key=metrics.get)

        ax.set_title(
            f"{monthly.index[0].strftime('%B')} {forecast_year}\nBest: {best_model}",
            fontsize=12 if duration.lower() != "all" else 10,
            fontweight="bold"
        )

        y_min = monthly.min().min()
        y_max = monthly.max().max()
        y_range = y_max - y_min

        pad = max(y_range * 0.12, 0.05 if plot_scale.lower() == "gwh" else 0.01)
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

        if plot_scale.lower() == "gwh":
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        else:
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))

        ax.grid(True, alpha=0.3)

    for j in range(len(selected_months), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{plot_title} for {country} ({forecast_year})",
        fontsize=18,
        y=0.985
    )

    first_visible_ax = next(ax for ax in axes if ax.get_visible())
    handles, labels = first_visible_ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
        handlelength=3.0,
        columnspacing=1.5
    )

    fig.supxlabel("Day of Month", fontsize=13)
    fig.supylabel(y_label, fontsize=13)

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.96])

    if save:
        file_path = os.path.join(
            output_dir,
            f"{country}_monthly_forecast_panels_{duration}_{plot_scale}_{forecast_year}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()
    
# Monthly RMSE heatmap
# This function creates a heatmap visualization of the monthly RMSE values for different forecasting models. 
# It pivots the input monthly_table to have months as rows and models as columns, and then uses seaborn to create a heatmap. 
# The function also allows for saving the resulting figure to a specified output directory.   

def plot_monthly_rmse_heatmap(
    monthly_table,
    country,
    forecast_year,
    output_dir="outputs/figures",
    save=True
):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    heatmap_df = monthly_table.pivot(
        index="Month",
        columns="Model",
        values="RMSE (log)"
    )

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    heatmap_df = heatmap_df.reindex(month_order)

    desired_model_order = [
        "SARIMAX Full",
        "SARIMAX Restricted",
        "Seasonal Naïve",
        "XGBoost No Weather",
        "XGBoost Full",
        "XGBoost Restricted"
    ]

    heatmap_df = heatmap_df.reindex(
        columns=[col for col in desired_model_order if col in heatmap_df.columns]
    )

    rename_map = {
        "SARIMAX Full": "SARIMAX\nFull",
        "SARIMAX Restricted": "SARIMAX\nRestricted",
        "Seasonal Naïve": "Seasonal\nNaïve",
        "XGBoost No Weather": "XGBoost\nNo Weather",
        "XGBoost Full": "XGBoost\nFull",
        "XGBoost Restricted": "XGBoost\nRestricted"
    }

    heatmap_df = heatmap_df.rename(columns=rename_map)

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd_r",
        linewidths=0.5,
        cbar_kws={"label": "RMSE"}
    )

    plt.ylabel("Month")
    plt.xlabel("Model")
    plt.tight_layout()

    if save:
        file_path = os.path.join(
            output_dir,
            f"{country}_monthly_rmse_heatmap_{forecast_year}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {file_path}")

    plt.show()