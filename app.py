# app.py

import streamlit as st
import pandas as pd
import torch
import numpy as np
from pipeline import (
    fetch_history,
    add_technical_indicators, 
    build_sequences_cols, 
    TCNForecast, 
    scaler_p, 
    pruned_feats, 
    seq_len,
    config
)

st.title("📈 Stock Forecast with Pruned TCN")

symbol = st.sidebar.text_input("Ticker", "AAPL")
if st.sidebar.button("Predict"):
    # 1. Fetch recent history
    df = fetch_history(symbol, '2014-01-01', 'today')  # or your own loader
    df_ind = add_technical_indicators(df)

    # 2. Build last test sequences
    X, y, _ = build_sequences_cols(df_ind, seq_len, pruned_feats, scaler=scaler_p)
    X_last = torch.tensor(X[-1:], dtype=torch.float32)
    
    # 3. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNForecast(
        input_size=X.shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(torch.load("models/tcn_pruned.pt", map_location=device))
    model.eval()
    
    # 4. Predict & denormalize
    with torch.no_grad():
        pred_norm = model(X_last.to(device)).cpu().item()
    close_min = scaler_p.data_min_[pruned_feats.index('Close')]
    close_max = scaler_p.data_max_[pruned_feats.index('Close')]
    pred_price = pred_norm * (close_max - close_min) + close_min
    
    st.metric("Next‐Day Predicted Close", f"${pred_price:.2f}")
