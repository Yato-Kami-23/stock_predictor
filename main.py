from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import save_model

from stock_predictor.data_fetcher import fetch_history
from stock_predictor.preprocess import prepare_data
from stock_predictor.model_def import build_lstm_model
from stock_predictor.train_utils import train_model
from stock_predictor.predict_utils import predict_on_test, forecast_future


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Stock Price Predictor (LSTM + yfinance)")
	parser.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol, e.g., AAPL")
	parser.add_argument("--period", type=str, default=None, help="yfinance period, e.g., 1y, 2y, 5y")
	parser.add_argument("--predict_days", type=int, default=None, help="Number of future days to predict")
	parser.add_argument("--window", type=int, default=60, help="Sequence window size")
	parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
	parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data for test set")
	parser.add_argument("--model_path", type=str, default="stock_model.h5", help="Path to save trained model")
	return parser.parse_args()


def prompt_if_needed(args: argparse.Namespace) -> argparse.Namespace:
	if not args.ticker:
		args.ticker = input("Enter stock ticker (e.g., AAPL): ").strip()
	if not args.period:
		args.period = input("Enter period (e.g., 1y, 2y, 5y): ").strip()
	if args.predict_days is None:
		args.predict_days = int(input("Enter number of prediction days (e.g., 5): ").strip())
	return args


def evaluate_and_plot(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	title: str,
	ticker: str,
) -> None:
	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	rmse = np.sqrt(mse)
	print(f"MAE: {mae:.4f}")
	print(f"RMSE: {rmse:.4f}")

	plt.figure(figsize=(10, 5))
	plt.plot(y_true, label="Actual", color="tab:blue")
	plt.plot(y_pred, label="Predicted", color="tab:orange")
	plt.title(f"{title} - {ticker}")
	plt.xlabel("Time (test samples)")
	plt.ylabel("Price")
	plt.legend()
	plt.tight_layout()
	plt.show()


def main() -> None:
	args = parse_args()
	args = prompt_if_needed(args)

	print("\n=== 1) Downloading data ===")
	data = fetch_history(args.ticker, args.period)
	print(f"Downloaded {len(data)} rows for {args.ticker} ({args.period}).")

	print("\n=== 2) Preparing data ===")
	X_train, y_train, X_test, y_test, scaler, last_window = prepare_data(
		dataframe=data,
		feature_col="Close",
		window_size=args.window,
		test_ratio=args.test_ratio,
	)
	print(
		f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}"
	)

	print("\n=== 3) Building model ===")
	model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
	model.summary()

	print("\n=== 4) Training model ===")
	history = train_model(
		model,
		X_train,
		y_train,
		X_val=X_test,
		y_val=y_test,
		epochs=args.epochs,
		batch_size=args.batch_size,
		patience=5,
		verbose=1,
	)
	print("Training complete.")

	print("\n=== 5) Evaluating on test set ===")
	y_test_true = scaler.inverse_transform(y_test).ravel()
	y_test_pred = predict_on_test(model, X_test, scaler)
	print(f"Test samples: {len(y_test_true)}")

	print("\n=== 6) Plotting Actual vs Predicted (Test) ===")
	evaluate_and_plot(y_test_true, y_test_pred, title="Actual vs Predicted (Test)", ticker=args.ticker)

	print("\n=== 7) Forecasting future days ===")
	_, future_forecasts = forecast_future(model, last_window, scaler, args.predict_days)
	print(f"Next {args.predict_days} day(s) forecast:")
	for i, val in enumerate(future_forecasts, start=1):
		print(f"Day +{i}: {val:.2f}")

	print(f"\n=== 8) Saving model to {args.model_path} ===")
	model.save(args.model_path)
	print("Model saved.")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nInterrupted.")
		sys.exit(1)
