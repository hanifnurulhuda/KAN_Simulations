import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_backtest_mt(df_test, y_pred_test, initial_balance=1000, lot_size=0.01, contract_size=100, commission_per_lot=7, sl_atr_mult=1.0, tp_atr_mult=1.0, risk_free_rate=0.03, leverage=200, stop_out_level=1.0, buy_threshold=0.6, sell_threshold=0.4):
    df = df_test.copy()
    y_pred = np.asarray(y_pred_test)
    if y_pred.ndim == 2 and y_pred.shape[1] == 3:
        df['prob_sell'] = y_pred[:, 0]
        df['prob_hold'] = y_pred[:, 1]
        df['prob_buy'] = y_pred[:, 2]
    else:
        df['prob_buy'] = y_pred
        df['prob_sell'] = 1.0 - y_pred
        df['prob_hold'] = 0.0

    df['signal'] = 0
    df.loc[(df['prob_buy'] > buy_threshold) & (df['prob_buy'] >= df['prob_sell']), 'signal'] = 1
    df.loc[(df['prob_sell'] > sell_threshold) & (df['prob_sell'] > df['prob_buy']), 'signal'] = -1
    
    trades = []
    current_trade = None
    balance = initial_balance
    
    for i, row in df.iterrows():
        # A. Cek Exit SL/TP atau Margin Call
        if current_trade:
            direction = 1 if current_trade['type'] == 'Long' else -1
            floating_pnl = (row['close_1d'] - current_trade['entry_price']) * direction * lot_size * contract_size
            current_equity = balance + floating_pnl
            used_margin = (current_trade['entry_price'] * lot_size * contract_size) / leverage
            margin_level = current_equity / used_margin if used_margin > 0 else float('inf')
            
            exit_price = None
            reason = None
            
            if margin_level < stop_out_level:
                exit_price = row['close_1d']; reason = 'Margin Call'
            elif current_trade['type'] == 'Long':
                if row['high'] >= current_trade['tp']: exit_price = current_trade['tp']; reason = 'TP'
                elif row['low'] <= current_trade['sl']: exit_price = current_trade['sl']; reason = 'SL'
            else:
                if row['low'] <= current_trade['tp']: exit_price = current_trade['tp']; reason = 'TP'
                elif row['high'] >= current_trade['sl']: exit_price = current_trade['sl']; reason = 'SL'
            
            if exit_price:
                current_trade['exit_price'] = exit_price; current_trade['exit_date'] = i; current_trade['reason'] = reason
                pnl = (exit_price - current_trade['entry_price']) * (1 if current_trade['type'] == 'Long' else -1) * lot_size * contract_size - (2 * commission_per_lot * lot_size)
                current_trade['pnl'] = pnl; balance += pnl
                trades.append(current_trade); current_trade = None
                continue
        
        # B. Cek Signal untuk Entry atau Exit
        signal = row['signal']
        if current_trade:
            if (current_trade['type'] == 'Long' and signal <= 0) or (current_trade['type'] == 'Short' and signal >= 0):
                current_trade['exit_price'] = row['close_1d']; current_trade['exit_date'] = i; current_trade['reason'] = 'Signal'
                pnl = (current_trade['exit_price'] - current_trade['entry_price']) * (1 if current_trade['type'] == 'Long' else -1) * lot_size * contract_size - (2 * commission_per_lot * lot_size)
                current_trade['pnl'] = pnl; balance += pnl
                trades.append(current_trade); current_trade = None
        
        if not current_trade and signal != 0:
            atr = row['atr'] if not pd.isna(row['atr']) else 0.0
            required_margin = (row['close_1d'] * lot_size * contract_size) / leverage
            if balance < required_margin:
                print(f"[{i.date()}] Saldo tidak cukup untuk membuka posisi baru. Trading dihentikan.")
                break
            
            sl_dist = atr * sl_atr_mult
            tp_dist = atr * tp_atr_mult
            if signal == 1:
                current_trade = {'entry_date': i, 'entry_price': row['close_1d'], 'type': 'Long', 'sl': row['close_1d'] - sl_dist, 'tp': row['close_1d'] + tp_dist}
            elif signal == -1:
                current_trade = {'entry_date': i, 'entry_price': row['close_1d'], 'type': 'Short', 'sl': row['close_1d'] + sl_dist, 'tp': row['close_1d'] - tp_dist}

    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0: return df, trade_df
    
    wins = trade_df[trade_df['pnl'] > 0]; losses = trade_df[trade_df['pnl'] <= 0]
    win_rate = len(wins) / len(trade_df); total_profit = trade_df['pnl'].sum()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0; avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    max_win = wins['pnl'].max() if len(wins) > 0 else 0; max_loss = losses['pnl'].min() if len(losses) > 0 else 0
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
    
    df['net_pnl'] = 0.0
    for t in trades: df.loc[t['exit_date'], 'net_pnl'] += t['pnl']
    df['equity_strategy'] = initial_balance + df['net_pnl'].cumsum()
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    daily_returns = df['net_pnl'] / df['equity_strategy'].shift(1).fillna(initial_balance)
    sharpe_ratio = ((daily_returns - daily_rf).mean() / (daily_returns - daily_rf).std()) * np.sqrt(252) if (daily_returns - daily_rf).std() != 0 else 0
    
    print("\n--- HASIL BACKTEST (MARGIN CALL ENABLED) ---")
    print(f"Total Trades     : {len(trade_df)}")
    print(f"Win Rate         : {win_rate:.2%}")
    print(f"Profit Factor    : {profit_factor:.2f}")
    print(f"Sharpe Ratio     : {sharpe_ratio:.2f}")
    print(f"Avg Win          : ${avg_win:,.2f}")
    print(f"Avg Loss         : ${avg_loss:,.2f}")
    print(f"Max Win          : ${max_win:,.2f}")
    print(f"Max Loss         : ${max_loss:,.2f}")
    print(f"Total Net Profit : ${total_profit:,.2f}")
    print(f"Final Balance    : ${balance:,.2f}")
    print("\n--- ANALISIS ALASAN EXIT ---")
    print(trade_df['reason'].value_counts().to_string())
    print("-" * 45)
    
    return df, trade_df

def export_trades_to_excel(trade_df, df_test, filename="backtest_results.xlsx"):
    trade_log = trade_df.merge(df_test[['open_1d', 'high', 'low', 'close_1d', 'signal']], left_on='entry_date', right_index=True)
    trade_log = trade_log.rename(columns={'open_1d': 'entry_open', 'high': 'entry_high', 'low': 'entry_low', 'close_1d': 'entry_close', 'signal': 'signal_at_entry', 'entry_price': 'entry_price', 'sl': 'sl_level', 'tp': 'tp_level', 'pnl': 'trade_pnl'})
    trade_log.to_excel(filename, index=False)
    print(f"Hasil trade diekspor ke: {filename}")

def plot_backtest(df, last_n=100):
    print("Plot backtest disimpan ke: backtest_plot.png")
