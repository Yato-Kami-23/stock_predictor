from __future__ import annotations

from typing import Tuple

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
	input_shape: Tuple[int, int],
	n_units: int = 64,
	dropout_rate: float = 0.2,
	learning_rate: float = 1e-3,
) -> Sequential:
	"""Build and compile a simple LSTM model for univariate forecasting.

	Args:
		input_shape: (timesteps, features), e.g., (window_size, 1).
		n_units: Number of LSTM units.
		dropout_rate: Dropout rate after LSTM.
		learning_rate: Learning rate for Adam.
	"""
	model = Sequential()
	model.add(LSTM(n_units, input_shape=input_shape, return_sequences=False))
	model.add(Dropout(dropout_rate))
	model.add(Dense(32, activation="relu"))
	model.add(Dense(1))

	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
	return model
