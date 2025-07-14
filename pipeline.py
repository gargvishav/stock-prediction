# pipeline.py

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Configuration --------------------
config = {
    'tickers': ['AAPL', 'MSFT', 'GOOGL'],
    'start_date': '2014-01-01',
    'end_date': '2024-12-31',
    'seq_len': 60,
    'batch_size': 32,
    'epochs': 20,
    'lr': 1e-3,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# Full feature list and pruned feature list
feature_cols = ['Open','High','Low','Close','Volume','SMA14','RSI14','MACD_hist']
pruned_feats = ['Open','High','Low','Close','Volume','MACD_hist']
seq_len = config['seq_len']

# -------------------- Data Utilities --------------------

def fetch_history(ticker, start, end):
    """
    Download historical OHLCV data for a ticker.
    Returns a DataFrame indexed by Date.
    """
    df = yf.download(ticker, start=start, end=end)
    return df


def fetch_new_bar(ticker):
    """
    Fetch the latest trading day bar (OHLCV) for a ticker.
    Returns a single-row DataFrame.
    """
    df = yf.download(ticker, period="2d")
    return df.tail(1)


# -------------------- Feature Engineering --------------------

def add_technical_indicators(df):
    """
    Add SMA14, RSI14, and MACD histogram to OHLCV DataFrame.
    Drops initial rows with NaNs after calculation.
    """
    df = df.copy()
    # 14-day Simple Moving Average
    df['SMA14'] = SMAIndicator(close=df['Close'], window=14).sma_indicator()
    # 14-day RSI
    df['RSI14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    # MACD histogram
    macd = MACD(close=df['Close'])
    df['MACD_hist'] = macd.macd_diff()
    return df.dropna()


# -------------------- Sequence Builder --------------------

def build_sequences_cols(df, seq_len, feature_cols, scaler=None):
    """
    Given a DataFrame and a list of feature columns:
    - Fits or reuses a MinMaxScaler to scale features.
    - Builds sliding-window sequences of length seq_len.
    Returns X (n_samples, seq_len, n_features), y (n_samples,), and the scaler.
    """
    feats = df[feature_cols]
    if scaler is None:
        scaler = MinMaxScaler().fit(feats)
        scaled = scaler.transform(feats)
    else:
        scaled = scaler.transform(feats)
    X, y = [], []
    close_idx = feature_cols.index('Close')
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, close_idx])
    return np.array(X), np.array(y), scaler


# -------------------- PyTorch Dataset --------------------

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sliding-window sequences.
    Automatically squeezes target y to shape (batch,).
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if self.y.ndim == 2 and self.y.size(1) == 1:
            self.y = self.y.squeeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Training & Evaluation --------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += loss_fn(preds, y_batch).item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# -------------------- Model Definitions --------------------

class GRUForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUForecast, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out.squeeze()


class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze()


class TCNForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 kernel_size=2, dilation=1):
        super(TCNForecast, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=kernel_size,
                               padding=(kernel_size-1)*dilation,
                               dilation=dilation)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        c = self.conv1(x)
        c = self.relu(c)
        c = self.dropout(c)
        out = c[:, :, -1]
        return self.fc(out).squeeze()


def get_model(name, input_size, config):
    if name == 'gru':
        return GRUForecast(input_size,
                           config['hidden_size'],
                           config['num_layers'],
                           config['dropout'])
    elif name == 'lstm':
        return LSTMForecast(input_size,
                            config['hidden_size'],
                            config['num_layers'],
                            config['dropout'])
    elif name == 'tcn':
        return TCNForecast(input_size,
                           config['hidden_size'],
                           config['num_layers'],
                           config['dropout'])
    else:
        raise ValueError(f"Unknown model name: {name}")


# -------------------- Continual Learning --------------------

# These globals will be initialized by your script or notebook
train_df = None        # DataFrame with enriched features
scaler_p  = None       # Fitted MinMaxScaler on pruned_feats
model_p   = None       # Pruned TCN model instance
optimizer_p = None     # Optimizer for fine-tuning

# Continual learning parameters
CONTINUAL_WINDOW_DAYS = 504
FINE_TUNE_EPOCHS     = 1
FINE_TUNE_BATCH_SIZE = config['batch_size']
FINE_TUNE_LR         = 1e-4
REPLAY_BUFFER_SIZE   = 500

# Replay buffer
replay_X = None
replay_y = None


def fine_tune_on_new_data(new_bar_df):
    """
    Append the new day's indicators, update replay buffer, and fine-tune the pruned TCN.
    new_bar_df: single-row DataFrame with OHLCV and indicator columns.
    """
    global train_df, scaler_p, model_p, optimizer_p, replay_X, replay_y

    # 1) Append new day and trim window
    train_df = pd.concat([train_df, new_bar_df]).iloc[-CONTINUAL_WINDOW_DAYS:]

    # 2) Rebuild sequences for pruned features and update scaler
    X_full, y_full, scaler_p = build_sequences_cols(
        train_df, seq_len, pruned_feats, scaler=None)
    # New day's latest sequence
    X_new = X_full[-1:].astype(np.float32)
    y_new = y_full[-1:].astype(np.float32)

    # 3) Update replay buffer
    if replay_X is None:
        # initialize buffer with random subset
        indices = np.random.choice(len(X_full),
                                   min(REPLAY_BUFFER_SIZE, len(X_full)),
                                   replace=False)
        replay_X = X_full[indices]
        replay_y = y_full[indices]
    replay_X = np.concatenate([replay_X, X_new], axis=0)[-REPLAY_BUFFER_SIZE:]
    replay_y = np.concatenate([replay_y, y_new], axis=0)[-REPLAY_BUFFER_SIZE:]

    # 4) Prepare mixed batch for fine-tuning
    mix_X = np.concatenate([X_new] * FINE_TUNE_BATCH_SIZE +
                           [replay_X[:FINE_TUNE_BATCH_SIZE]], axis=0)
    mix_y = np.concatenate([y_new] * FINE_TUNE_BATCH_SIZE +
                           [replay_y[:FINE_TUNE_BATCH_SIZE]], axis=0)
    mix_X, mix_y = shuffle(mix_X, mix_y)
    ds = SequenceDataset(mix_X, mix_y)
    dl = DataLoader(ds, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=True)

    # 5) Fine-tune
    model_p.train()
    for _ in range(FINE_TUNE_EPOCHS):
        train_one_epoch(model_p, dl, nn.MSELoss(), optimizer_p, torch.device('cpu'))

    print("✅ Fine-tuned on new data for one day.")
