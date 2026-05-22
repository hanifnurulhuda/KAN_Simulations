import contextlib
import itertools
import os

import pandas as pd

import backtest_utils
import data_utils
import model_utils


def build_predictions():
    df = data_utils.fetch_data()
    df = data_utils.compute_bollinger(df)
    df = data_utils.compute_atr(df)
    X, Y, df_ready = data_utils.create_dataset(df)

    X_train, Y_train, X_test, _ = data_utils.split_dataset(X, Y, train_ratio=0.8)
    model, X_mean, X_std = model_utils.prepare_model(X_train)
    X_train_norm = model_utils.normalize_data(X_train, X_mean, X_std)
    X_test_norm = model_utils.normalize_data(X_test, X_mean, X_std)

    model_utils.train_model(model, X_train_norm, Y_train)
    y_pred_test = model_utils.get_predictions(model, X_test_norm)
    df_test = df_ready.iloc[len(X_train):].copy()
    return df_test, y_pred_test


def summarize_trades(trade_df, initial_balance=1000):
    if trade_df.empty:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'net_profit': 0.0,
            'final_balance': initial_balance,
        }

    wins = trade_df[trade_df['pnl'] > 0]
    losses = trade_df[trade_df['pnl'] <= 0]
    gross_profit = wins['pnl'].sum()
    gross_loss = losses['pnl'].sum()
    net_profit = trade_df['pnl'].sum()

    return {
        'trades': len(trade_df),
        'win_rate': len(wins) / len(trade_df),
        'profit_factor': abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf'),
        'net_profit': net_profit,
        'final_balance': initial_balance + net_profit,
    }


def main():
    df_test, y_pred_test = build_predictions()
    results = []

    grid = {
        'buy_threshold': [0.35, 0.40, 0.45, 0.50],
        'sell_threshold': [0.30, 0.35, 0.40, 0.45],
        'sl_atr_mult': [0.8, 1.0, 1.2, 1.5],
        'tp_atr_mult': [1.0, 1.5, 2.0, 2.5],
    }

    keys = list(grid.keys())
    with open(os.devnull, 'w') as devnull:
        for values in itertools.product(*(grid[key] for key in keys)):
            params = dict(zip(keys, values))
            with contextlib.redirect_stdout(devnull):
                _, trade_df = backtest_utils.run_backtest_mt(df_test, y_pred_test, **params)

            results.append({**params, **summarize_trades(trade_df)})

    result_df = pd.DataFrame(results).sort_values(
        by=['final_balance', 'profit_factor', 'trades'],
        ascending=[False, False, False],
    )
    result_df.to_csv('optimization_results.csv', index=False)

    print("\n--- TOP 10 OPTIMIZATION RESULTS ---")
    print(result_df.head(10).to_string(index=False))
    print("\nHasil lengkap disimpan ke: optimization_results.csv")


if __name__ == "__main__":
    main()
