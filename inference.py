import torch
from kan import KAN
import data_utils
import model_utils
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

def load_deployed_model(checkpoint_dir="./model"):
    # Initialize with 8 features
    model = KAN(width=[8, 1], grid=10, k=3, device="cpu")
    
    try:
        path = f"{checkpoint_dir}/0.0"
        state_dict = torch.load(f'{path}_state', map_location='cpu')
        model.load_state_dict(state_dict)
        model.cache_data = torch.load(f'{path}_cache_data', map_location='cpu')
        
        print(f"Model berhasil dimuat dari {checkpoint_dir}")
        return model
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None

def run_inference():
    # 1. Fetch data
    print("Fetching latest data for inference...")
    df = data_utils.fetch_data()
    df = data_utils.compute_bollinger(df)
    df = data_utils.compute_atr(df)
    
    # 2. Preprocess
    _, _, df_ready = data_utils.create_dataset(df)
    
    # Ambil data terbaru (terakhir)
    latest_data = df_ready.iloc[[-1]]
    features = ['bb_pct_scaled', 'return_lag_1', 'return_lag_2', 
                'rolling_mean_return_5', 'rolling_vol_5', 'ratio_to_1w', 'ratio_to_1m', 'trend_slope']
    
    X = torch.tensor(latest_data[features].values, dtype=torch.float32)
    
    # 3. Load Model & Stats
    model = load_deployed_model()
    X_mean, X_std = model_utils.load_normalization_stats()
    
    if model is None:
        return

    # 4. Predict
    # Apply normalization using saved stats
    X_norm = model_utils.normalize_data(X, X_mean, X_std)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(X_norm)).item()
    
    status = "Buy" if pred > 0.45 else "Sell" if pred < 0.45 else "Hold"
    
    print(f"\nPrediksi Terbaru: Probabilitas Buy = {pred:.4f} -> {status}")
    
    if status != "Hold":
        price = latest_data['close_1d'].item()
        atr = latest_data['atr'].item()
        sl_distance = atr * 1.0
        tp_distance = atr * 1.0
        
        if status == "Buy":
            entry = price
            sl = entry - sl_distance
            tp = entry + tp_distance
        else: # Sell
            entry = price
            sl = entry + sl_distance
            tp = entry - tp_distance
            
        print(f"--- Trading Plan (Dinamis ATR) ---")
        print(f"Entry      : {entry:.2f}")
        print(f"Stop Loss  : {sl:.2f}")
        print(f"Take Profit: {tp:.2f}")

if __name__ == "__main__":
    run_inference()
