"""
prepare.py — Fixed evaluation harness (DO NOT MODIFY)

Generates synthetic SPY-like daily price data with embedded patterns
that a neural network can learn to exploit. Provides train/val split,
feature engineering, trading simulation, and evaluation.

The synthetic data has these learnable signals:
- Mean reversion after large moves (>2% daily move tends to reverse)
- Volume spikes often precede reversals
- Short-term momentum (3-day trend tends to continue for 1 more day)
- Volatility clustering (high vol days cluster together)
"""

import numpy as np
import time

# ---- Constants ----
TIME_BUDGET = 10  # seconds per experiment
SEED = 42
N_DAYS = 600       # total trading days (~2.4 years)
TRAIN_RATIO = 0.7  # 70% train, 30% validation
INITIAL_PRICE = 450.0  # starting SPY price
INITIAL_CAPITAL = 10000.0  # $10,000 starting capital

# ---- Data Generation ----
def _generate_prices():
    """Generate synthetic SPY-like daily OHLCV data with learnable patterns."""
    rng = np.random.RandomState(SEED)

    prices = [INITIAL_PRICE]
    volumes = []
    high_prices = []
    low_prices = []

    for day in range(1, N_DAYS):
        prev = prices[-1]

        # Base random return (daily ~0.05% drift, ~1% vol)
        base_return = rng.normal(0.0003, 0.01)

        # Pattern 1: Mean reversion after big moves
        if len(prices) >= 2:
            last_return = (prices[-1] - prices[-2]) / prices[-2]
            if abs(last_return) > 0.02:
                # After a >2% move, tend to reverse ~60% of the time
                base_return -= last_return * 0.3

        # Pattern 2: 3-day momentum
        if len(prices) >= 4:
            recent_returns = [(prices[-i] - prices[-i-1]) / prices[-i-1] for i in range(1, 4)]
            if all(r > 0 for r in recent_returns):
                base_return += 0.003  # momentum continuation
            elif all(r < 0 for r in recent_returns):
                base_return -= 0.003

        # Pattern 3: Volatility clustering
        if len(prices) >= 2:
            last_return = (prices[-1] - prices[-2]) / prices[-2]
            if abs(last_return) > 0.015:
                base_return *= 1.5  # amplify moves after big moves

        # Apply return
        new_price = prev * (1 + base_return)
        prices.append(max(new_price, 1.0))  # floor at $1

        # Generate volume (higher on big move days)
        base_vol = rng.uniform(50_000_000, 120_000_000)
        vol_spike = 1.0 + abs(base_return) * 30  # volume spikes on big moves
        # Pattern 4: Volume spike predicts next-day reversal
        if rng.random() < 0.15:
            vol_spike *= 2.5  # random volume spike
        volumes.append(int(base_vol * vol_spike))

        # High/Low
        intra_range = abs(base_return) + rng.uniform(0.002, 0.008)
        high_prices.append(new_price * (1 + intra_range / 2))
        low_prices.append(new_price * (1 - intra_range / 2))

    # First day
    volumes.insert(0, int(rng.uniform(60_000_000, 100_000_000)))
    high_prices.insert(0, INITIAL_PRICE * 1.003)
    low_prices.insert(0, INITIAL_PRICE * 0.997)

    return (np.array(prices), np.array(high_prices),
            np.array(low_prices), np.array(volumes, dtype=float))

def _compute_features(prices, highs, lows, volumes):
    """Compute trading features from price/volume data."""
    n = len(prices)
    features = {}

    # Daily returns
    returns = np.zeros(n)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    features['return_1d'] = returns

    # 3-day return
    ret_3d = np.zeros(n)
    ret_3d[3:] = (prices[3:] - prices[:-3]) / prices[:-3]
    features['return_3d'] = ret_3d

    # 5-day return
    ret_5d = np.zeros(n)
    ret_5d[5:] = (prices[5:] - prices[:-5]) / prices[:-5]
    features['return_5d'] = ret_5d

    # Moving average ratios
    for window in [5, 10, 20]:
        ma = np.convolve(prices, np.ones(window)/window, mode='full')[:n]
        ma[:window] = prices[:window]
        features[f'ma_ratio_{window}'] = prices / ma - 1.0

    # Volatility (5-day rolling std of returns)
    vol = np.zeros(n)
    for i in range(5, n):
        vol[i] = np.std(returns[i-5:i])
    features['volatility_5d'] = vol

    # Volume change (relative to 10-day average)
    vol_ma = np.convolve(volumes, np.ones(10)/10, mode='full')[:n]
    vol_ma[:10] = volumes[:10]
    vol_ma[vol_ma == 0] = 1  # avoid division by zero
    features['volume_ratio'] = volumes / vol_ma - 1.0

    # RSI-like (14-day)
    rsi = np.full(n, 50.0)
    for i in range(14, n):
        gains = np.maximum(returns[i-14:i], 0)
        losses = np.maximum(-returns[i-14:i], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    features['rsi'] = (rsi - 50.0) / 50.0  # normalize to [-1, 1]

    # High-low range (intraday volatility)
    hl_range = (highs - lows) / prices
    features['hl_range'] = hl_range

    return features

def get_data():
    """
    Generate features and labels for SPY direction prediction.

    Returns:
        x_train: (n_train, n_features) feature matrix
        y_train: (n_train, 1) labels — 1.0 if next day up, 0.0 if down
        x_val:   (n_val, n_features)
        y_val:   (n_val, 1)
        prices_train: raw prices for train period
        prices_val:   raw prices for validation period
    """
    prices, highs, lows, volumes = _generate_prices()
    features = _compute_features(prices, highs, lows, volumes)

    # Build feature matrix (skip first 20 days for indicator warmup)
    start = 20
    feature_names = sorted(features.keys())
    n_samples = len(prices) - start - 1  # -1 because we need next day for label

    X = np.zeros((n_samples, len(feature_names)))
    for j, name in enumerate(feature_names):
        X[:, j] = features[name][start:start + n_samples]

    # Labels: 1 if next day price goes up, 0 if down
    y = np.zeros((n_samples, 1))
    for i in range(n_samples):
        day_idx = start + i
        if prices[day_idx + 1] > prices[day_idx]:
            y[i] = 1.0
        else:
            y[i] = 0.0

    # Normalize features (z-score)
    split = int(n_samples * TRAIN_RATIO)
    train_mean = X[:split].mean(axis=0)
    train_std = X[:split].std(axis=0)
    train_std[train_std == 0] = 1.0
    X = (X - train_mean) / train_std

    x_train, y_train = X[:split], y[:split]
    x_val, y_val = X[split:], y[split:]

    prices_train = prices[start:start + split + 1]
    prices_val = prices[start + split:start + n_samples + 1]

    return x_train, y_train, x_val, y_val, prices_train, prices_val

def get_feature_names():
    """Return the list of feature names in order."""
    prices, highs, lows, volumes = _generate_prices()
    features = _compute_features(prices, highs, lows, volumes)
    return sorted(features.keys())

# ---- Evaluation ----
def evaluate(model):
    """
    Evaluate a trained model by simulating trading on the validation set.

    The model should output a value per day:
        > 0.5 → BUY (go long)
        <= 0.5 → SELL (go short or stay out)

    Prints val_profit (%) and val_accuracy (% correct direction).
    Returns val_profit.
    """
    _, _, x_val, y_val, _, prices_val = get_data()

    # Get predictions
    predictions = model(x_val)
    signals = (predictions > 0.5).astype(float).flatten()  # 1 = buy, 0 = sell/out
    actual_up = y_val.flatten()

    # Accuracy
    correct = (signals == actual_up).sum()
    accuracy = correct / len(actual_up) * 100

    # Simulate trading
    capital = INITIAL_CAPITAL
    position = 0  # 0 = no position, 1 = long
    entry_price = 0.0
    trades = 0

    for i in range(len(signals)):
        price_today = prices_val[i]
        price_tomorrow = prices_val[i + 1] if i + 1 < len(prices_val) else price_today

        if signals[i] == 1:  # BUY signal
            if position == 0:
                position = 1
                entry_price = price_today
                trades += 1
            # Hold through the day, gain/lose based on next day
            daily_return = (price_tomorrow - price_today) / price_today
            capital *= (1 + daily_return)
        else:  # SELL/OUT signal
            if position == 1:
                position = 0
            # No position, capital unchanged

    profit_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print(f"val_profit={profit_pct:.2f}")
    print(f"val_accuracy={accuracy:.1f}")
    print(f"val_trades={trades}")
    print(f"val_final_capital=${capital:.2f}")

    return profit_pct

# ---- Timer ----
class Timer:
    """Enforce the time budget."""
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def remaining(self):
        return max(0, TIME_BUDGET - self.elapsed())

    def expired(self):
        return self.elapsed() >= TIME_BUDGET
