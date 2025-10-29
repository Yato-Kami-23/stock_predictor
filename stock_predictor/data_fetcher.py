from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(ticker: str, period: str) -> pd.DataFrame:
	"""Fetch historical OHLCV data for a given ticker and period using yfinance.

	Args:
		ticker: Stock symbol, e.g., "AAPL".
		period: yfinance period (e.g., "1y", "2y", "5y", "max").

	Returns:
		DataFrame with DatetimeIndex and at least the "Close" column.
	"""
	if not ticker or not isinstance(ticker, str):
		raise ValueError("ticker must be a non-empty string")
	if not period or not isinstance(period, str):
		raise ValueError("period must be a non-empty string")

	data = yf.download(ticker.strip().upper(), period=period, auto_adjust=True, progress=False)
	if data is None or data.empty:
		raise RuntimeError(f"No data returned for ticker={ticker} period={period}")

	# Keep consistent ordering and clean index
	data = data.sort_index()
	data = data[~data.index.duplicated(keep="first")]

	# Ensure Close exists and handle missing values conservatively
	if "Close" not in data.columns:
		raise RuntimeError("Expected 'Close' column in downloaded data")

	close = data[["Close"]].copy()
	close["Close"] = close["Close"].ffill().bfill()
	close = close.dropna()

	return close
