## Stock Price Predictor (LSTM + yfinance)

This project predicts future stock closing prices using an LSTM neural network trained on historical data from `yfinance`.

### Features
- **Streamlit Web Interface**: Interactive UI for easy use (default)
- **Selectable Stock Ticker**: Choose from popular tickers or enter a custom symbol
- Preprocess: handle missing values, scale features, and create sequences.
- LSTM model (TensorFlow/Keras) with early stopping and model saving.
- Evaluate with RMSE and MAE.
- Visualize actual vs predicted prices with interactive charts.

### Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Streamlit Web Interface (Recommended)
Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser. Configure parameters in the sidebar:
- **Stock Ticker**: Pick from a dropdown or choose "Custom..." to type any ticker
- **Time Period**: Select historical data period (1y, 2y, 5y, max)
- **Prediction Days**: Number of future days to predict (1-30)
- **Model Parameters**: Adjust window size, epochs, batch size, and test ratio
- Click **"Train & Predict"** to start

#### Command Line Interface
Run the predictor and follow prompts:

```bash
python main.py
```

You will be asked for:
- Stock ticker (e.g., `AAPL`)
- Time period (e.g., `1y`, `5y`) — same semantics as `yfinance` periods
- Number of prediction days (e.g., `5`)

Alternatively, you can pass CLI arguments:

```bash
python main.py --ticker AAPL --period 2y --predict_days 7 --window 60 --epochs 25
```

The trained model is saved to `stock_model.h5`.

### Project Structure
- `app.py`: **Streamlit web interface** (recommended). Interactive UI with a ticker dropdown and custom entry.
- `main.py`: CLI entry point. Coordinates fetching, preprocessing, training, evaluation, plotting, and saving the model.
- `stock_predictor/data_fetcher.py`: Download data from `yfinance`.
- `stock_predictor/preprocess.py`: Clean, scale, create supervised sequences, split train/test.
- `stock_predictor/model_def.py`: Build and compile the LSTM model.
- `stock_predictor/train_utils.py`: Training utilities (early stopping, history handling).
- `stock_predictor/predict_utils.py`: Inference helpers for test-set predictions and multi-day forecasting.

### Notes
- Plots will display in a window. If running headless, set the Matplotlib backend accordingly or save figures.
- This is a baseline; tune hyperparameters and architecture for better performance.
