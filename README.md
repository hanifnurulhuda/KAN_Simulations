# KAN Simulations

KAN Simulations is a Python research project for testing a Kolmogorov-Arnold Network (KAN) trading model on XAUUSD multi-timeframe data. The pipeline loads offline Excel market data, builds technical and Fibonacci-based features, trains a three-class model, evaluates Buy/Hold/Sell predictions, runs a MetaTrader-style backtest, and searches for better backtest parameters.

This project is intended for strategy research and simulation only. It does not place live trades.

## Main Features

- Offline XAUUSD data loading from local Excel files.
- Multi-timeframe context using H1 as the base timeframe with H4 and D1 resampled into the H1 index.
- Technical feature engineering with Bollinger Bands, ATR, returns, volatility, trend slope, and Fibonacci retracement context.
- Three-class KAN classifier with dedicated outputs for Sell, Hold, and Buy.
- Weighted cross-entropy training to reduce class imbalance issues and improve Sell detection.
- MetaTrader-style backtest with lot size, contract size, commission, leverage, margin call logic, ATR stop loss, and ATR take profit.
- Parameter grid optimization for Buy threshold, Sell threshold, stop-loss ATR multiplier, and take-profit ATR multiplier.
- Inference script for the latest available row in the offline dataset.
- Exported trade logs, optimization results, and plots.

## Project Structure

```text
KAN_Simulation/
├── simulasi_kan.py          # Main training, evaluation, backtest, and plotting script
├── inference.py             # Loads the saved model and predicts the latest signal
├── optimize_backtest.py     # Grid search optimizer for backtest parameters
├── data_utils.py            # Data loading, indicators, features, labels, and dataset split
├── model_utils.py           # KAN model setup, training, prediction, evaluation, and save/load helpers
├── backtest_utils.py        # MetaTrader-style backtest and Excel export
├── plot_utils.py            # Prediction, metric, and loss plots
├── XAUUSD_H1.xlsx           # Offline H1 data
├── XAUUSD_H4.xlsx           # Offline H4 data
├── XAUUSD_D1.xlsx           # Offline D1 data
├── model/                   # Saved model checkpoint and normalization statistics
├── backtest_results.xlsx    # Exported backtest trade log
└── optimization_results.csv # Full optimizer result table
```

## Data Inputs

The project expects these Excel files in the repository root:

- `XAUUSD_H1.xlsx`
- `XAUUSD_H4.xlsx`
- `XAUUSD_D1.xlsx`

Each file should contain at least these columns:

- `time`
- `open`
- `high`
- `low`
- `close`

The loader renames them internally to `Datetime`, `Open`, `High`, `Low`, and `Close`.

## Feature Engineering

The current model uses the following feature columns:

- `bb_pct_scaled`: scaled Bollinger Band position.
- `return_lag_1`: previous return.
- `return_lag_2`: return from two bars ago.
- `rolling_mean_return_5`: five-bar rolling mean return.
- `rolling_vol_5`: five-bar rolling return volatility.
- `ratio_to_1w`: log ratio between H1 close and resampled H4 close.
- `ratio_to_1m`: log ratio between H1 close and resampled D1 close.
- `trend_slope`: linear regression slope over a rolling price window.
- `fib_position`: close position inside the rolling high-low Fibonacci range.
- `dist_fib_382`: normalized distance from the 38.2% Fibonacci level.
- `dist_fib_500`: normalized distance from the 50.0% Fibonacci level.
- `dist_fib_618`: normalized distance from the 61.8% Fibonacci level.

## Target Labels

The model predicts three classes:

- `Sell`: encoded as `0.0`.
- `Hold`: encoded as `0.5`.
- `Buy`: encoded as `1.0`.

Labels are generated from forward returns and rolling return quantiles:

- Future return above the upper quantile becomes `Buy`.
- Future return below the lower quantile becomes `Sell`.
- Everything else becomes `Hold`.

## Model Design

The model is a KAN classifier with three output neurons:

```text
[input_features, 3]
```

The output order is:

```text
Sell, Hold, Buy
```

Training uses `CrossEntropyLoss` with class weights. This is important because the dataset can be imbalanced, and the earlier one-output regression approach tended to ignore Sell signals.

## Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install numpy pandas torch scipy matplotlib openpyxl pykan
```

Depending on the KAN package distribution, the import used by the project is:

```python
from kan import KAN
```

## Run The Full Simulation

```powershell
.venv\Scripts\python.exe simulasi_kan.py
```

This will:

- Load offline Excel data.
- Compute indicators and engineered features.
- Train the KAN classifier.
- Evaluate training and testing metrics.
- Run the backtest on unseen test data.
- Save the model and normalization statistics.
- Export the trade log to `backtest_results.xlsx`.
- Save plots:
  - `prediction_plot.png`
  - `metrics_plot.png`
  - `loss_plot.png`

## Run Latest Inference

```powershell
.venv\Scripts\python.exe inference.py
```

Example output:

```text
Prediksi Terbaru: Sell=0.3628, Hold=0.3494, Buy=0.2878 -> Hold
```

Inference loads:

- `model/0.0_state`
- `model/0.0_cache_data`
- `model/norm_stats.json`

## Run Backtest Optimization

```powershell
.venv\Scripts\python.exe optimize_backtest.py
```

The optimizer tests combinations of:

- `buy_threshold`
- `sell_threshold`
- `sl_atr_mult`
- `tp_atr_mult`

It saves the complete result table to:

```text
optimization_results.csv
```

## Recent Test Result

The latest tested three-class model produced the following testing metrics:

```text
Buy  Precision 0.1563 | Recall 0.4356 | F1 0.2300
Hold Precision 0.6791 | Recall 0.3065 | F1 0.4224
Sell Precision 0.3473 | Recall 0.4798 | F1 0.4030
Overall Accuracy: 0.3665
```

Default backtest result:

```text
Total Trades     : 278
Win Rate         : 67.99%
Profit Factor    : 2.42
Sharpe Ratio     : 2.03
Total Net Profit : $2,273.29
Final Balance    : $3,273.29
```

Best optimizer result from the current grid:

```text
buy_threshold  = 0.35
sell_threshold = 0.35
sl_atr_mult    = 1.5
tp_atr_mult    = 2.5
trades         = 477
win_rate       = 65.41%
profit_factor  = 2.31
net_profit     = $3,935.68
final_balance  = $4,935.68
```

## Important Notes

- Backtest performance is not a guarantee of future live trading results.
- The current data is offline and file-based, so live market changes are not included unless the Excel files are updated.
- The optimizer can overfit to the test period if used without walk-forward validation.
- Add out-of-sample testing, walk-forward validation, and stricter risk management before considering any real trading use.

## Suggested Next Improvements

- Add a `requirements.txt` file for reproducible installation.
- Add walk-forward validation instead of a single train/test split.
- Add confusion matrix export for Buy/Hold/Sell predictions.
- Add risk-per-trade position sizing instead of fixed lot size.
- Add spread and slippage modeling.
- Add drawdown, Sortino, Calmar, and expectancy metrics to the optimizer.
