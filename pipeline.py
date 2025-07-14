import pandas as pd
import numpy as np
import yfinance as yf
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

# Feature lists
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA14', 'RSI14', 'MACD_hist']
pruned_feats  = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_hist']
seq_len       = config['seq_len']

# -------------------- Data Utilities --------------------
from datetime import datetime

def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    if isinstance(end, str) and end.lower() == 'today':
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data for {ticker} from {start} to {end}.")
    return df


def fetch_new_bar(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2d")
    return df.tail(1)

# -------------------- Feature Engineering --------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve OHLCV and append SMA14, RSI14, MACD_hist.
    """
    df = df.copy()
    # SMA14
    df['SMA14'] = df['Close'].rolling(window=14, min_periods=14).mean()
    # RSI14
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD_hist
    ema12       = df['Close'].ewm(span=12, adjust=False).mean()
    ema26       = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - signal_line
    # Drop warm-up NaNs
    return df.dropna()

# -------------------- Sequence Builder --------------------

def build_sequences(df: pd.DataFrame,
                    seq_len: int,
                    feature_cols: list,
                    scaler: MinMaxScaler = None):
    """
    Build sliding-window sequences of length seq_len over feature_cols.
    If scaler=None, fits a MinMaxScaler; else reuses it.
    Returns X (n, seq_len, n_features), y (n,), scaler.
    """
    # 1) Drop NaNs in features
    df_clean = df.dropna(subset=feature_cols)
    features = df_clean[feature_cols]
    # 2) Scale
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
    else:
        scaled = scaler.transform(features)
    # 3) Slide windows
    X, y = [], []
    close_idx = feature_cols.index('Close')
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, close_idx])
    return np.array(X), np.array(y), scaler

# -------------------- PyTorch Dataset --------------------

class SequenceDataset(Dataset):
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
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze()

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()

class TCNForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 kernel_size=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size,
                               kernel_size=kernel_size,
                               padding=(kernel_size-1)*dilation,
                               dilation=dilation)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0,2,1)
        c = self.conv1(x)
        c = self.relu(c)
        c = self.dropout(c)
        return self.fc(c[:,:, -1]).squeeze()


def get_model(name, input_size, config):
    if name == 'gru':
        return GRUForecast(input_size, config['hidden_size'],
                           config['num_layers'], config['dropout'])
    if name == 'lstm':
        return LSTMForecast(input_size, config['hidden_size'],
                             config['num_layers'], config['dropout'])
    if name == 'tcn':
        return TCNForecast(input_size, config['hidden_size'],
                            config['num_layers'], config['dropout'])
    raise ValueError(f"Unknown model: {name}")

# -------------------- Continual Learning --------------------

# Globals to be initialized externally
train_df    = None
scaler_p    = None
model_p     = None
optimizer_p = None

CONTINUAL_WINDOW_DAYS = 504
FINE_TUNE_EPOCHS     = 1
FINE_TUNE_BATCH_SIZE = config['batch_size']
FINE_TUNE_LR         = 1e-4
REPLAY_BUFFER_SIZE   = 500

replay_X = None
replay_y = None

def fine_tune_on_new_data(new_bar_df: pd.DataFrame):
    global train_df, scaler_p, model_p, optimizer_p, replay_X, replay_y

    # 1) Append and trim window
    train_df = pd.concat([train_df, new_bar_df]).iloc[-CONTINUAL_WINDOW_DAYS:]

    # 2) Rebuild sequences (fits fresh scaler)
    X_full, y_full, scaler_p = build_sequences(train_df, seq_len, pruned_feats, scaler=None)
    X_new = X_full[-1:].astype(np.float32)
    y_new = y_full[-1:].astype(np.float32)

    # 3) Update replay buffer
    if replay_X is None:
        idx = np.random.choice(len(X_full), min(REPLAY_BUFFER_SIZE, len(X_full)), replace=False)
        replay_X = X_full[idx]
        replay_y = y_full[idx]
    replay_X = np.concatenate([replay_X, X_new], axis=0)[-REPLAY_BUFFER_SIZE:]
    replay_y = np.concatenate([replay_y, y_new], axis=0)[-REPLAY_BUFFER_SIZE:]

    # 4) Prepare mixed batch
    mix_X = np.concatenate([X_new] * FINE_TUNE_BATCH_SIZE + [replay_X[:FINE_TUNE_BATCH_SIZE]], axis=0)
    mix_y = np.concatenate([y_new] * FINE_TUNE_BATCH_SIZE + [replay_y[:FINE_TUNE_BATCH_SIZE]], axis=0)
    mix_X, mix_y = shuffle(mix_X, mix_y)
    ds = SequenceDataset(mix_X, mix_y)
    dl = DataLoader(ds, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=True)

    # 5) Fine-tune
    model_p.train()
    for _ in range(FINE_TUNE_EPOCHS):
        train_one_epoch(model_p, dl, nn.MSELoss(), optimizer_p, torch.device('cpu'))

    print("✅ Fine-tuned on new data for one day.")
