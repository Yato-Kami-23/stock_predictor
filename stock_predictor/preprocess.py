from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_sequences(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Create supervised learning sequences for univariate time series.

	Given scaled values of shape (n, 1), return X of shape (n-window, window, 1)
	and y of shape (n-window, 1), where each X[i] is a sliding window.
	"""
	X, y = [], []
	for i in range(window_size, len(values)):
		X.append(values[i - window_size : i])
		y.append(values[i])
	return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(
	dataframe: pd.DataFrame,
	feature_col: str = "Close",
	window_size: int = 60,
	test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, np.ndarray]:
	"""Prepare data for LSTM training and evaluation.

	- Fits scaler on the training segment only, then transforms the full series
	- Builds supervised sequences
	- Splits X/y into train/test using the original chronological boundary

	Returns X_train, y_train, X_test, y_test, scaler, last_window
	"""
	if feature_col not in dataframe.columns:
		raise ValueError(f"Column '{feature_col}' not found in dataframe")

	values = dataframe[[feature_col]].astype(float).values
	n = len(values)
	if n < window_size + 10:
		raise ValueError(
			f"Not enough data ({n} rows) for window_size={window_size}. Try a longer period."
		)

	train_end = int(n * (1.0 - test_ratio))
	train_end = max(train_end, window_size + 1)

	scaler = MinMaxScaler(feature_range=(0.0, 1.0))
	scaler.fit(values[:train_end])
	scaled_all = scaler.transform(values)

	X_all, y_all = create_sequences(scaled_all, window_size)
	# Sequences start after 'window_size'; align split accordingly
	seq_split_index = train_end - window_size
	seq_split_index = max(seq_split_index, 1)

	X_train = X_all[:seq_split_index]
	y_train = y_all[:seq_split_index]
	X_test = X_all[seq_split_index:]
	y_test = y_all[seq_split_index:]

	last_window = scaled_all[-window_size:]
	return X_train, y_train, X_test, y_test, scaler, last_window
