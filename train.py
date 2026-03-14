"""
train.py — The mutable training script (agent edits this file)

Trains a neural network to predict next-day SPY direction (up/down)
and generate buy/sell trading signals.

Features available (from prepare.py):
- return_1d, return_3d, return_5d: recent price returns
- ma_ratio_5, ma_ratio_10, ma_ratio_20: moving average ratios
- volatility_5d: recent volatility
- volume_ratio: volume relative to average
- rsi: relative strength index (normalized)
- hl_range: intraday high-low range
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import prepare

# ---- Hyperparameters ----
HIDDEN_SIZE = 16
NUM_LAYERS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
ACTIVATION = "tanh"  # tanh, relu, sigmoid

# ---- Activation Functions ----
def activation_fn(x):
    if ACTIVATION == "tanh":
        return np.tanh(x)
    elif ACTIVATION == "relu":
        return np.maximum(0, x)
    elif ACTIVATION == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return np.tanh(x)

def activation_derivative(x):
    if ACTIVATION == "tanh":
        t = np.tanh(x)
        return 1 - t ** 2
    elif ACTIVATION == "relu":
        return (x > 0).astype(float)
    elif ACTIVATION == "sigmoid":
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)
    t = np.tanh(x)
    return 1 - t ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# ---- Model ----
class TradingNet:
    def __init__(self, n_features):
        rng = np.random.RandomState(42)
        self.weights = []
        self.biases = []

        # Hidden layers
        in_size = n_features
        for _ in range(NUM_LAYERS):
            w = rng.randn(in_size, HIDDEN_SIZE) * np.sqrt(2.0 / in_size)
            b = np.zeros((1, HIDDEN_SIZE))
            self.weights.append(w)
            self.biases.append(b)
            in_size = HIDDEN_SIZE

        # Output layer (single neuron -> sigmoid -> probability of UP)
        w = rng.randn(in_size, 1) * np.sqrt(2.0 / in_size)
        b = np.zeros((1, 1))
        self.weights.append(w)
        self.biases.append(b)

    def forward(self, x):
        """Forward pass with sigmoid output for binary classification."""
        self.pre_activations = []
        self.activations = [x]

        h = x
        for i in range(NUM_LAYERS):
            z = h @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            h = activation_fn(z)
            self.activations.append(h)

        # Output layer with sigmoid
        z = h @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(z)
        out = sigmoid(z)
        self.activations.append(out)
        return out

    def __call__(self, x):
        return self.forward(x)

    def backward(self, y_true):
        """Backpropagation for binary cross-entropy loss."""
        m = y_true.shape[0]
        grads_w = []
        grads_b = []

        # Output gradient (sigmoid + BCE)
        pred = self.activations[-1]
        delta = (pred - y_true) / m

        for i in range(len(self.weights) - 1, -1, -1):
            dw = self.activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                delta = delta @ self.weights[i].T
                delta = delta * activation_derivative(self.pre_activations[i - 1])

        return grads_w, grads_b

    def update(self, grads_w, grads_b, lr):
        """SGD update."""
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

# ---- Training ----
def train():
    x_train, y_train, _, _, _, _ = prepare.get_data()
    timer = prepare.Timer()
    n_features = x_train.shape[1]
    model = TradingNet(n_features)

    epoch = 0
    while not timer.expired():
        # Shuffle
        perm = np.random.permutation(len(x_train))
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        # Mini-batch SGD
        for start in range(0, len(x_train), BATCH_SIZE):
            if timer.expired():
                break
            x_batch = x_shuffled[start:start + BATCH_SIZE]
            y_batch = y_shuffled[start:start + BATCH_SIZE]

            model.forward(x_batch)
            grads_w, grads_b = model.backward(y_batch)
            model.update(grads_w, grads_b, LEARNING_RATE)

        epoch += 1

    print(f"training_epochs={epoch}")
    print(f"training_seconds={timer.elapsed():.1f}")
    return model

# ---- Main ----
if __name__ == "__main__":
    model = train()
    prepare.evaluate(model)