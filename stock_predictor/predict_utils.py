from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def predict_on_test(model, X_test, scaler: MinMaxScaler):
	"""Predict y for the test set and inverse-transform to original scale.
	Returns predictions as 1D ndarray.
	"""
	y_pred_scaled = model.predict(X_test, verbose=0)
	y_pred = scaler.inverse_transform(y_pred_scaled)
	return y_pred.ravel()


def forecast_future(
	model,
	last_window_scaled: np.ndarray,
	scaler: MinMaxScaler,
	n_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Recursive multi-step forecast using the last scaled window.

	Args:
		last_window_scaled: shape (window, 1) scaled window.
		n_days: number of steps to forecast.

	Returns:
		scaled_forecasts, forecasts (both shape (n_days,))
	"""
	window = last_window_scaled.copy()
	forecasts_scaled = []
	for _ in range(n_days):
		input_seq = window.reshape(1, window.shape[0], window.shape[1])
		next_scaled = model.predict(input_seq, verbose=0)
		forecasts_scaled.append(next_scaled[0, 0])
		window = np.vstack([window[1:], next_scaled])

	forecasts_scaled = np.array(forecasts_scaled, dtype=float).reshape(-1, 1)
	forecasts = scaler.inverse_transform(forecasts_scaled).ravel()
	return forecasts_scaled.ravel(), forecasts
