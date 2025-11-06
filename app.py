from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stock_predictor.data_fetcher import fetch_history
from stock_predictor.preprocess import prepare_data
from stock_predictor.model_def import build_lstm_model
from stock_predictor.train_utils import train_model
from stock_predictor.predict_utils import predict_on_test, forecast_future

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# --- Global styles ---
st.markdown(
    """
    <style>
      :root {
        --brand-bg: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0ea5e9 100%);
        --card-bg: rgba(255, 255, 255, 0.06);
        --card-border: rgba(255, 255, 255, 0.12);
        --accent: #0ea5e9;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
      }

      /* App hero header */
      .app-hero {
        background: var(--brand-bg);
        padding: 28px 22px;
        border-radius: 16px;
        color: #e5f2ff;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.4);
        margin-bottom: 16px;
      }
      .app-hero h1 { margin: 0 0 6px 0; font-size: 28px; font-weight: 800; }
      .app-hero p { margin: 0; opacity: 0.9; }

      /* KPI card styling */
      .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
      .kpi-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 16px 14px;
        backdrop-filter: blur(6px);
      }
      .kpi-label { font-size: 12px; letter-spacing: .3px; opacity: .85; }
      .kpi-value { font-size: 22px; font-weight: 800; color: #ffffff; }

      /* Beautify default metrics */
      div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 12px 14px;
      }
      div[data-testid="stMetricValue"] { font-weight: 800; }

      /* Buttons */
      .stButton>button {
        background: linear-gradient(180deg, #38bdf8 0%, #0ea5e9 100%);
        color: #0b1220;
        border: none;
        font-weight: 700;
        padding: 10px 14px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(14, 165, 233, .35);
      }
      .stButton>button:hover { filter: brightness(1.05); transform: translateY(-1px); }

      /* Expanders & charts */
      details {
        background: rgba(2, 6, 23, .04);
        border-radius: 12px;
        border: 1px solid rgba(2, 6, 23, .08);
      }

      /* Sidebar accent */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        border-right: 1px solid rgba(255,255,255,.06);
      }
      [data-testid="stSidebar"] * { color: #dbeafe !important; }

      /* Subtle hr */
      .soft-hr { height: 1px; background: rgba(148, 163, 184, .25); margin: 8px 0 16px; }

      /* Footer */
      .footer {
        opacity: .75;
        font-size: 13px;
        border-top: 1px dashed rgba(148,163,184,.35);
        padding-top: 10px;
        margin-top: 18px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-hero">
      <h1>📈 Stock Price Predictor (LSTM)</h1>
      <p>Train an LSTM on historical prices, evaluate performance, and forecast future values — in minutes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
	st.header("Configuration")
	# Ticker selection
	predefined = [
		"AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "NVDA", "META", "NFLX", "AMD", "INTC", "SPY",
		"Custom...",
	]
	ticker_choice = st.selectbox(
		"Stock Ticker",
		options=predefined,
		index=0,
		help="Choose a ticker or select Custom to type your own"
	)
	if ticker_choice == "Custom...":
		custom_ticker = st.text_input("Enter custom ticker", value="AAPL").strip().upper()
		selected_ticker = custom_ticker if custom_ticker else "AAPL"
	else:
		selected_ticker = ticker_choice
	st.markdown(f"**Selected Ticker:** {selected_ticker}")
	
	period = st.selectbox(
		"Time Period",
		options=["1y", "2y", "5y", "max"],
		index=1,
		help="Historical data period to fetch"
	)
	
	predict_days = st.number_input(
		"Prediction Days",
		min_value=1,
		max_value=30,
		value=5,
		help="Number of future days to predict"
	)
	
	st.markdown("---")
	st.subheader("Model Parameters")
	
	window_size = st.slider(
		"Window Size",
		min_value=30,
		max_value=120,
		value=60,
		step=10,
		help="Sequence window size for LSTM"
	)
	
	epochs = st.slider(
		"Training Epochs",
		min_value=5,
		max_value=50,
		value=25,
		help="Number of training epochs"
	)
	
	batch_size = st.selectbox(
		"Batch Size",
		options=[16, 32, 64],
		index=1,
		help="Training batch size"
	)
	
	test_ratio = st.slider(
		"Test Set Ratio",
		min_value=0.1,
		max_value=0.4,
		value=0.2,
		step=0.05,
		help="Fraction of data for testing"
	)
	
	run_button = st.button("🚀 Train & Predict", type="primary", use_container_width=True)

if run_button:
	progress_bar = st.progress(0)
	status_text = st.empty()
	
	try:
		# Step 1: Download data
		status_text.text("📥 Downloading historical data...")
		progress_bar.progress(10)
		data = fetch_history(selected_ticker, period)
		st.success(f"✅ Downloaded {len(data)} rows for {selected_ticker} ({period})")
		
		# Display data preview
		with st.expander("📊 Historical Data Preview", expanded=False):
			st.dataframe(data.tail(10), use_container_width=True)
			st.line_chart(data["Close"])
		
		# Step 2: Prepare data
		status_text.text("🔧 Preparing data...")
		progress_bar.progress(30)
		
		try:
			X_train, y_train, X_test, y_test, scaler, last_window = prepare_data(
				dataframe=data,
				feature_col="Close",
				window_size=window_size,
				test_ratio=test_ratio,
			)
		except ValueError as e:
			st.error(f"❌ Error preparing data: {e}")
			st.stop()
		
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Training Samples", X_train.shape[0])
		with col2:
			st.metric("Test Samples", X_test.shape[0])
		with col3:
			st.metric("Window Size", window_size)
		with col4:
			st.metric("Features", X_train.shape[2])
		
		# Step 3: Build model
		status_text.text("🏗️ Building LSTM model...")
		progress_bar.progress(50)
		
		model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
		
		with st.expander("📐 Model Architecture", expanded=False):
			model_summary = []
			model.summary(print_fn=lambda x: model_summary.append(x))
			st.text("\n".join(model_summary))
		
		# Step 4: Train model
		status_text.text("🎓 Training model...")
		progress_bar.progress(60)
		
		history_placeholder = st.empty()
		with history_placeholder.container():
			history = train_model(
				model,
				X_train,
				y_train,
				X_val=X_test,
				y_val=y_test,
				epochs=epochs,
				batch_size=batch_size,
				patience=5,
				verbose=0,  # Suppress verbose output for Streamlit
			)
		
		history_placeholder.empty()
		progress_bar.progress(80)
		
		# Plot training history
		fig_history, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
		
		ax1.plot(history.history["loss"], label="Training Loss", color="blue")
		ax1.plot(history.history["val_loss"], label="Validation Loss", color="red")
		ax1.set_title("Model Loss")
		ax1.set_xlabel("Epoch")
		ax1.set_ylabel("Loss (MSE)")
		ax1.legend()
		ax1.grid(True, alpha=0.3)
		
		ax2.plot(history.history["mae"], label="Training MAE", color="blue")
		ax2.plot(history.history["val_mae"], label="Validation MAE", color="red")
		ax2.set_title("Model MAE")
		ax2.set_xlabel("Epoch")
		ax2.set_ylabel("MAE")
		ax2.legend()
		ax2.grid(True, alpha=0.3)
		
		plt.tight_layout()
		st.pyplot(fig_history)
		plt.close()
		
		# Step 5: Evaluate on test set
		status_text.text("📊 Evaluating model...")
		progress_bar.progress(85)
		
		y_test_true = scaler.inverse_transform(y_test).ravel()
		y_test_pred = predict_on_test(model, X_test, scaler)
		
		mae = mean_absolute_error(y_test_true, y_test_pred)
		mse = mean_squared_error(y_test_true, y_test_pred)
		rmse = np.sqrt(mse)
		
		# Display metrics
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("MAE", f"${mae:.2f}")
		with col2:
			st.metric("RMSE", f"${rmse:.2f}")
		with col3:
			st.metric("Test Samples", len(y_test_true))
		
		# Plot actual vs predicted
		fig_test, ax = plt.subplots(figsize=(12, 6))
		ax.plot(y_test_true, label="Actual", color="tab:blue", linewidth=2)
		ax.plot(y_test_pred, label="Predicted", color="tab:orange", linewidth=2)
		ax.set_title(f"Actual vs Predicted Prices (Test Set) - {selected_ticker}", fontsize=14, fontweight="bold")
		ax.set_xlabel("Time (test samples)", fontsize=12)
		ax.set_ylabel("Price ($)", fontsize=12)
		ax.legend(fontsize=11)
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		st.pyplot(fig_test)
		plt.close()
		
		# Step 6: Forecast future days
		status_text.text("🔮 Forecasting future prices...")
		progress_bar.progress(95)
		
		_, future_forecasts = forecast_future(model, last_window, scaler, predict_days)
		
		# Display forecasts
		st.subheader("📅 Future Price Forecasts")
		
		forecast_df = pd.DataFrame({
			"Day": [f"Day +{i}" for i in range(1, predict_days + 1)],
			"Predicted Price ($)": future_forecasts
		})
		
		st.dataframe(forecast_df, use_container_width=True, hide_index=True)
		
		# Plot future forecasts
		fig_future, ax = plt.subplots(figsize=(12, 6))
		
		# Plot historical data
		historical_prices = data["Close"].values[-60:]  # Last 60 days
		historical_days = range(len(historical_prices))
		ax.plot(historical_days, historical_prices, label="Historical", color="tab:blue", linewidth=2)
		
		# Plot predictions
		future_days = range(len(historical_prices), len(historical_prices) + predict_days)
		ax.plot(future_days, future_forecasts, label="Forecasted", color="tab:orange", linewidth=2, marker="o")
		
		# Add vertical line separator
		ax.axvline(x=len(historical_prices) - 1, color="gray", linestyle="--", alpha=0.5, label="Today")
		
		ax.set_title(f"Stock Price Forecast - {selected_ticker}", fontsize=14, fontweight="bold")
		ax.set_xlabel("Days", fontsize=12)
		ax.set_ylabel("Price ($)", fontsize=12)
		ax.legend(fontsize=11)
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		st.pyplot(fig_future)
		plt.close()
		
		# Step 7: Save model
		status_text.text("💾 Saving model...")
		progress_bar.progress(100)
		
		model.save("stock_model.h5")
		st.success("✅ Model saved as `stock_model.h5`")
		
		status_text.empty()
		progress_bar.empty()
		
		st.balloons()
		st.markdown(
			"""
			<div class="footer">
			  Built with ❤️ using Streamlit, Keras, and yfinance. Tweak parameters from the sidebar and iterate quickly.
			</div>
			""",
			unsafe_allow_html=True,
		)
		
	except Exception as e:
		st.error(f"❌ Error: {str(e)}")
		st.exception(e)
		status_text.empty()
		progress_bar.empty()

else:
	st.info("👈 Configure the parameters in the sidebar and click 'Train & Predict' to start!")
	
	# Show example
	st.markdown("### 📖 How to Use")
	st.markdown("""
	1. **Select Time Period**: Choose how much historical data to fetch (1y, 2y, 5y, or max)
	2. **Set Prediction Days**: Number of future days you want to predict (1-30 days)
	3. **Adjust Model Parameters**: Tune window size, epochs, batch size, and test ratio
	4. **Click Train & Predict**: The model will train and generate predictions
	5. **View Results**: See metrics, plots, and future forecasts
	
	**Tip**: Pick a ticker from the dropdown or choose "Custom..." and type any valid symbol (e.g., TSLA, NVDA).
	""")

