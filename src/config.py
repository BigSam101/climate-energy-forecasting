COUNTRY_CODE_MAP = {
    "Norway": "NO",
    "Sweden": "SE",
    "Finland": "FI"
}


def get_country_settings(country):
    code = COUNTRY_CODE_MAP[country]
    target_col = f"log_load_{country}"
    lag_prefix = code.lower()

    return code, target_col, lag_prefix


def get_exog_specs(country):
    code, target_col, lag_prefix = get_country_settings(country)

    full_exog_vars = [
        f'HDD_{country}',
        f'Extreme_Cold_{code}',
        f'HDD_Extreme_{code}',
        f'CDD_{country}',
        f'Extreme_Warm_{code}',
        f'CDD_Extreme_{code}',
        f'lag1_log_{lag_prefix}',
        f'Holiday_{country}'
    ]

    restricted_exog_vars = [
        f'HDD_{country}',
        f'Extreme_Cold_{code}',
        f'HDD_Extreme_{code}',
        f'lag1_log_{lag_prefix}',
        f'Holiday_{country}'
    ]

    specs = {
        "Full": full_exog_vars,
        "Restricted": restricted_exog_vars
    }

    return specs