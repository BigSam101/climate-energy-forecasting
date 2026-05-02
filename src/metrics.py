import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def compute_metrics(y_true, y_pred, use_mape=False):
    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    }

    if use_mape:
        metrics["MAPE (%)"] = mape(y_true, y_pred)

    return metrics