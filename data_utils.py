import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress

FEATURE_COLUMNS = [
    'bb_pct_scaled',
    'return_lag_1',
    'return_lag_2',
    'rolling_mean_return_5',
    'rolling_vol_5',
    'ratio_to_1w',
    'ratio_to_1m',
    'trend_slope',
    'fib_position',
    'dist_fib_382',
    'dist_fib_500',
    'dist_fib_618',
]

def load_excel_data(file_path):
    # Sesuaikan kolom sesuai hasil inspeksi (time, open, high, low, close)
    df = pd.read_excel(file_path)
    df = df.rename(columns={
        'time': 'Datetime',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    })
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df

def fetch_data():
    # Load local excel files
    h1 = load_excel_data("XAUUSD_H1.xlsx")
    h4 = load_excel_data("XAUUSD_H4.xlsx")
    d1 = load_excel_data("XAUUSD_D1.xlsx")

    # Base: H1
    df = pd.DataFrame(index=h1.index)
    df['close_1d'] = h1['Close']
    df['open_1d'] = h1['Open']
    df['high'] = h1['High']
    df['low'] = h1['Low']
    
    # Sinkronisasi H4 dan D1 ke index H1 (ffill)
    h4_resampled = h4['Close'].reindex(df.index, method='ffill')
    d1_resampled = d1['Close'].reindex(df.index, method='ffill')
    
    df['close_1w'] = h4_resampled # Mapping H4 as 'weekly' context for model
    df['close_1m'] = d1_resampled # Mapping D1 as 'monthly' context for model
    
    return df

def compute_bollinger(df, period=20, std_dev=2):
    prices = df['close_1d'].values
    n = len(prices)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        w = prices[i - period + 1 : i + 1]
        m = w.mean()
        s = w.std()
        mid[i] = m
        upper[i] = m + std_dev * s
        lower[i] = m - std_dev * s

    band = upper - lower
    pct_b = np.divide(prices - lower, band,
                      out=np.full(n, 0.5),
                      where=np.isfinite(band) & (band != 0))
    df['bb_pct'] = np.where(np.isfinite(pct_b), np.clip(pct_b, 0.0, 1.0), 0.5)
    df['bb_upper'] = upper
    df['bb_mid'] = mid
    df['bb_lower'] = lower
    return df

def compute_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close_1d']
    h_l = high - low
    h_c = abs(high - close.shift(1))
    l_c = abs(low - close.shift(1))
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=period).mean()
    return df

def compute_linear_regression_slope(prices, period=20):
    slopes = np.full(len(prices), np.nan)
    x = np.arange(period)
    for i in range(period, len(prices) + 1):
        y = prices[i - period : i]
        slope, _, _, _, _ = linregress(x, y)
        slopes[i - 1] = slope
    return slopes

def compute_fibonacci_features(df, period=50):
    rolling_high = df['high'].rolling(window=period).max()
    rolling_low = df['low'].rolling(window=period).min()
    price_range = rolling_high - rolling_low
    safe_range = price_range.replace(0, np.nan)

    fib_382 = rolling_high - (price_range * 0.382)
    fib_500 = rolling_high - (price_range * 0.500)
    fib_618 = rolling_high - (price_range * 0.618)

    df['fib_position'] = (df['close_1d'] - rolling_low) / safe_range
    df['dist_fib_382'] = (df['close_1d'] - fib_382) / safe_range
    df['dist_fib_500'] = (df['close_1d'] - fib_500) / safe_range
    df['dist_fib_618'] = (df['close_1d'] - fib_618) / safe_range
    return df

def create_dataset(df, horizon=1, upper_p=0.85, lower_p=0.25):
    # 1. Feature Engineering
    df['return_1d'] = df['close_1d'].pct_change()
    df['return_lag_1'] = df['return_1d'].shift(1)
    df['return_lag_2'] = df['return_1d'].shift(2)
    df['rolling_mean_return_5'] = df['return_1d'].rolling(window=5).mean()
    df['rolling_vol_5'] = df['return_1d'].rolling(window=5).std()
    df['ratio_to_1w'] = np.log(df['close_1d'] / (df['close_1w'] + 1e-8))
    df['ratio_to_1m'] = np.log(df['close_1d'] / (df['close_1m'] + 1e-8))
    df['bb_pct_scaled'] = (df['bb_pct'] - 0.5) * 2.0
    df['trend_slope'] = compute_linear_regression_slope(df['close_1d'].values, period=20)
    df = compute_fibonacci_features(df, period=50)

    # 2. Target Labeling (Forward-looking)
    df['future_return'] = df['close_1d'].pct_change(periods=horizon).shift(-horizon)
    df['hist_rolling_return'] = df['close_1d'].pct_change(periods=horizon)
    df['upper_threshold'] = df['hist_rolling_return'].rolling(window=250).quantile(upper_p)
    df['lower_threshold'] = df['hist_rolling_return'].rolling(window=250).quantile(lower_p)

    def label_target(row):
        if pd.isna(row['future_return']) or pd.isna(row['upper_threshold']) or pd.isna(row['lower_threshold']):
            return np.nan
        if row['future_return'] > row['upper_threshold']:
            return 1.0  # Buy
        elif row['future_return'] < row['lower_threshold']:
            return 0.0  # Sell
        else:
            return 0.5  # Hold

    df['target'] = df.apply(label_target, axis=1)

    # 3. Cleanup & Final Tensors
    df_ready = df.dropna().copy()
    
    # Ensure a basic signal column exists for export
    df_ready['signal'] = df_ready['target'].apply(lambda x: 1 if x==1.0 else -1 if x==0.0 else 0)
    
    X = torch.tensor(df_ready[FEATURE_COLUMNS].values, dtype=torch.float32)
    Y = torch.tensor(df_ready['target'].values, dtype=torch.float32).reshape(-1, 1)

    return X, Y, df_ready

def split_dataset(X, Y, train_ratio=0.8):
    total_len = len(X)
    train_size = int(total_len * train_ratio)
    
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    return X_train, Y_train, X_test, Y_test
