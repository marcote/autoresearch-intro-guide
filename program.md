# AutoResearch: SPY Trading Signal Optimization

## Objective

You are an autonomous ML research agent. Your goal is to maximize the
validation profit (`val_profit`) of a neural network that predicts
next-day SPY price direction (up/down) and generates trading signals.

The model receives market features (returns, moving averages, RSI,
volume, volatility) and outputs a probability of the price going up.
If probability > 0.5 → BUY. Otherwise → stay out.

## Rules

1. You may ONLY modify `train.py`.
2. You may NOT modify `prepare.py` (it contains the fixed evaluation and data).
3. You may NOT install new packages. Only use `numpy`.
4. Training must complete within the TIME_BUDGET (10 seconds).
5. The metric to maximize is `val_profit` (% return on validation period).

## What you can change in train.py

- Network architecture: number of layers, neurons per layer
- Activation functions (relu, tanh, sigmoid, swish, etc.)
- Learning rate, batch size, number of epochs
- Weight initialization strategy
- Optimizer logic (SGD, momentum, Adam-like, learning rate schedules)
- Regularization (dropout, weight decay, L1/L2)
- Feature engineering or selection (which features to use, transformations)
- Signal threshold (doesn't have to be 0.5)
- Loss function modifications
- Any other training technique that fits in a single file

## What you should NOT change

- The `if __name__ == "__main__"` block structure
- The call to `prepare.evaluate(model)` at the end
- The output format (must print `val_profit=<value>`)
- The model must be callable: `model(x) -> predictions`

## Available Features (10 total)

From prepare.py, the feature matrix columns (in alphabetical order):
1. hl_range — intraday high-low range
2. ma_ratio_10 — price vs 10-day moving average
3. ma_ratio_20 — price vs 20-day moving average
4. ma_ratio_5 — price vs 5-day moving average
5. return_1d — yesterday's return
6. return_3d — 3-day return
7. return_5d — 5-day return
8. rsi — relative strength index (normalized)
9. volatility_5d — 5-day rolling volatility
10. volume_ratio — volume vs 10-day average

## Strategy Hints

- The data has embedded patterns: mean reversion, momentum, volume signals
- Think about which features might be most predictive
- Consider that financial data is noisy — regularization matters
- The buy/sell threshold doesn't have to be 0.5
- Sometimes simpler models generalize better than complex ones
- Consider using different subsets of features
