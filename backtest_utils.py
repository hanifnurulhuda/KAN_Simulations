import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_backtest_mt(df_test, y_pred_test, initial_balance=400, lot_size=0.1, contract_size=100, commission_per_lot=7, sl_atr_mult=1.5, tp_atr_mult=3.0, risk_free_rate=0.03):
    """
    Backtest simulasi MetaTrader (Lot-based) dengan Sharpe Ratio yang disesuaikan.
    """
    df = df_test.copy()
    df['prob_buy'] = y_pred_test
    
    # 1. Tentukan Signal
    df['signal'] = 0
    df.loc[df['prob_buy'] > 0.6, 'signal'] = 1
    df.loc[df['prob_buy'] < 0.4, 'signal'] = -1
    
    # 2. Iterasi untuk Simulasi Trade
    trades = []
    current_trade = None
    
    for i, row in df.iterrows():
        # A. Cek Exit SL/TP jika ada trade terbuka
        if current_trade:
            exit_price = None
            reason = None
            
            if current_trade['type'] == 'Long':
                if row['high'] >= current_trade['tp']:
                    exit_price = current_trade['tp']
                    reason = 'TP'
                elif row['low'] <= current_trade['sl']:
                    exit_price = current_trade['sl']
                    reason = 'SL'
            else: # Short
                if row['low'] <= current_trade['tp']:
                    exit_price = current_trade['tp']
                    reason = 'TP'
                elif row['high'] >= current_trade['sl']:
                    exit_price = current_trade['sl']
                    reason = 'SL'
            
            if exit_price:
                current_trade['exit_price'] = exit_price
                current_trade['exit_date'] = i
                current_trade['reason'] = reason
                trades.append(current_trade)
                current_trade = None
                continue # Trade ditutup, lanjut ke hari berikutnya
        
        # B. Cek Signal untuk Entry atau Exit berdasarkan Signal
        signal = row['signal']
        
        # Exit berdasarkan signal (jika belum ditutup SL/TP)
        if current_trade:
            # Jika sinyal berubah atau Netral, tutup trade
            if (current_trade['type'] == 'Long' and signal <= 0) or \
               (current_trade['type'] == 'Short' and signal >= 0):
                current_trade['exit_price'] = row['close_1d']
                current_trade['exit_date'] = i
                current_trade['reason'] = 'Signal'
                trades.append(current_trade)
                current_trade = None
        
        # Entry berdasarkan signal (jika tidak ada trade)
        if not current_trade and signal != 0:
            atr = row['atr']
            if pd.isna(atr): atr = 0.0 # Fallback
            
            sl_dist = atr * sl_atr_mult
            tp_dist = atr * tp_atr_mult
            
            if signal == 1: # Long
                current_trade = {
                    'entry_date': i,
                    'entry_price': row['close_1d'],
                    'type': 'Long',
                    'sl': row['close_1d'] - sl_dist,
                    'tp': row['close_1d'] + tp_dist
                }
            elif signal == -1: # Short
                current_trade = {
                    'entry_date': i,
                    'entry_price': row['close_1d'],
                    'type': 'Short',
                    'sl': row['close_1d'] + sl_dist,
                    'tp': row['close_1d'] - tp_dist
                }

    # 3. Hitung PnL per Trade
    trade_list = []
    for t in trades:
        price_diff = t['exit_price'] - t['entry_price']
        if t['type'] == 'Short': price_diff = -price_diff
        
        pnl = price_diff * lot_size * contract_size
        commission = 2 * (commission_per_lot * lot_size)
        net_pnl = pnl - commission
        t['pnl'] = net_pnl
        trade_list.append(t)

    trade_df = pd.DataFrame(trade_list)
    
    # 4. Metrik Performa
    wins = trade_df[trade_df['pnl'] > 0]
    losses = trade_df[trade_df['pnl'] <= 0]
    
    win_rate = len(wins) / len(trade_df) if len(trade_df) > 0 else 0
    total_profit = trade_df['pnl'].sum()
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
    
    # 5. Equity Curve
    df['net_pnl'] = 0.0
    for t in trades:
        df.loc[t['exit_date'], 'net_pnl'] += t['pnl']
    
    df['equity_strategy'] = initial_balance + df['net_pnl'].cumsum()
    
    # 6. Sharpe Ratio (Anualisasi dengan risk-free rate)
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    # Menggunakan daily_pnl dalam bentuk return terhadap balance
    daily_returns = df['net_pnl'] / df['equity_strategy'].shift(1).fillna(initial_balance)
    excess_daily_returns = daily_returns - daily_rf
    
    if excess_daily_returns.std() != 0:
        sharpe_ratio = (excess_daily_returns.mean() / excess_daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
        
    print("\n--- HASIL BACKTEST METATRADER (TRADE-BASED WITH SL/TP) ---")
    print(f"Total Trades     : {len(trade_df)}")
    print(f"Win Rate         : {win_rate:.2%}")
    print(f"Profit Factor    : {profit_factor:.2f}")
    print(f"Sharpe Ratio     : {sharpe_ratio:.2f} (RF={risk_free_rate:.1%})")
    print(f"Total Net Profit : ${total_profit:,.2f}")
    print(f"Final Balance    : ${df['equity_strategy'].iloc[-1]:,.2f}")
    print("-" * 45)
    
    return df, trade_df

def plot_backtest(df, last_n=100):
    df_slice = df.iloc[-last_n:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_slice.index, df_slice['equity_strategy'], label='KAN Strategy', color='blue', lw=2)
    plt.title(f"MetaTrader Style Backtest: Equity Curve (Last {last_n} Days)")
    plt.xlabel("Date")
    plt.ylabel("Balance ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("backtest_plot.png")
    print(f"Plot backtest disimpan ke: backtest_plot.png (Zoomed last {last_n} days)")
