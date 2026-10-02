from __future__ import annotations
from tensorflow.keras.layers import LSTM, Dense, Dropout

"""Task 1 — One-step time series forecasting with LSTM (Keras).

Students implement the core pipeline:
- make_windows
- time_split
- build_model
- train_model

Everything else (metrics, evaluation, plotting, demo) is provided.

The notebook for the module explains the theory and a reference workflow.
This starter is intentionally framework-light and focuses on correct shapes,
leakage-free time splits, and reproducible evaluation.
"""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ----------------------------
# Metrics (provided)
# ----------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predictions. Shape must match y_true.

    Returns
    -------
    float
        MAE value: mean(|y_true - y_pred|).

    Notes
    -----
    This function flattens inputs to 1D before computing the metric.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predictions. Shape must match y_true.

    Returns
    -------
    float
        RMSE value: sqrt(mean((y_true - y_pred)^2)).

    Notes
    -----
    This function flattens inputs to 1D before computing the metric.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ----------------------------
# Core pipeline (students implement)
# ----------------------------

def make_windows(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D time series into supervised windows for one-step forecasting.

    We build pairs for each time index t >= window:
        X[t] = series[t-window : t]
        y[t] = series[t]

    Parameters
    ----------
    series : np.ndarray
        One-dimensional time series of length T. Shape: (T,).
        Must contain numeric finite values.
    window : int
        Window length w (number of past time steps). Must satisfy:
        1 <= window < len(series).

    Returns
    -------
    X : np.ndarray
        Input windows with explicit feature dimension for Keras RNN layers.
        Shape: (N, window, 1), where N = T - window.
    y : np.ndarray
        Targets (next value after each window).
        Shape: (N, 1).

    Notes
    -----
    Keras RNN layers expect inputs shaped as (batch, time, features).
    Here features=1 because the time series is univariate.
    """
    series = np.array(series, dtype=np.float32) 
    if series.ndim == 1 and np.isfinite(series).all():
        if isinstance(window, int) and window >= 1 and window < len(series):
            future_windows = []
            answers = []
            for t in range(window, len(series)):
                future_windows.append(series[t - window : t])
                answers.append(series[t])
            X = np.array(future_windows, dtype=np.float32).reshape(-1,window,1)
            y = np.array(answers, dtype=np.float32).reshape(-1,1)
        else: 
            raise ValueError("Invalid input")
    else:
        raise ValueError("Invalid input") 
    return X,y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split windows into train/val/test using time order (NO shuffling).

    This prevents *data leakage* typical for time series when random shuffling is used.

    Parameters
    ----------
    X : np.ndarray
        Windowed inputs. Shape: (N, window, 1).
    y : np.ndarray
        Targets. Shape: (N, 1).
    train_frac : float
        Fraction of samples used for training, e.g. 0.70.
    val_frac : float
        Fraction of samples used for validation, e.g. 0.15.
        Test fraction is computed as 1 - train_frac - val_frac.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) : tuple
        Time-preserving splits.

    Raises
    ------
    ValueError
        If fractions are invalid, or any split becomes empty.

    Notes
    -----
    - Do NOT shuffle.
    - The split is performed on already-windowed samples.
    """
    if len(X) == len(y) and train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1:
        n = len(X)
        train_end = int(train_frac * n)
        val_end = int(n * (train_frac + val_frac))
        if train_end > 0 and val_end > train_end and val_end < n: 
            X_train, y_train =  X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
        else:
            raise ValueError("Empty part")
    else:
        raise ValueError("Error")
    return  (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(
    window: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile an LSTM model for one-step forecasting.

    Parameters
    ----------
    window : int
        Number of time steps in each input window. Model input shape: (window, 1).
    n_units : int
        Number of LSTM units.
    dense_units : int
        Units in the intermediate Dense layer.
    dropout : float
        Dropout rate applied after the LSTM layer.
    learning_rate : float
        Learning rate for Adam optimizer.

    Returns
    -------
    tf.keras.Model
        Compiled model with:
        - input shape:  (None, window, 1)
        - output shape: (None, 1)
        - loss: MSE
        - metric: MAE

    Notes
    -----
    You may change the architecture slightly, but keep I/O shapes the same.
    """
    model = tf.keras.Sequential([
        LSTM(n_units,input_shape=(window, 1)),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model



def train_model(
    series: np.ndarray,
    window: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray, tf.keras.callbacks.History]:
    """Train an LSTM model on a time series and return model + test split.

    Workflow
    --------
    1) Create windows: X, y = make_windows(series, window)
    2) Time-based split: (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(...)
    3) Build model: model = build_model(window, ...)
    4) Fit model on train with validation
    5) Return (model, X_test, y_test, history)

    Parameters
    ----------
    series : np.ndarray
        1D time series. Shape: (T,).
    window : int
        Window length.
    train_frac : float
        Train fraction.
    val_frac : float
        Validation fraction.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    seed : int
        Random seed for reproducibility.
    verbose : int
        Verbosity for model.fit.

    Returns
    -------
    model : tf.keras.Model
        Trained model.
    X_test : np.ndarray
        Test windows. Shape: (N_test, window, 1).
    y_test : np.ndarray
        Test targets. Shape: (N_test, 1).
    history : tf.keras.callbacks.History
        Keras training history.

    Notes
    -----
    - Use time-based split to avoid leakage.
    - Prefer using EarlyStopping (optional) to reduce overfitting.
    - Keep the function deterministic as much as possible.
    """
    tf.keras.utils.set_random_seed(seed)
    X, y = make_windows(series,window)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y, train_frac, val_frac)
    model = build_model(window)
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train,y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[callbacks],
        shuffle=False
    )
    return model, X_test, y_test, history



# ----------------------------
# Evaluation (provided)
# ----------------------------

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a trained model on test data and compute MAE & RMSE.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model returning predictions of shape (N, 1).
    X_test : np.ndarray
        Test windows. Shape: (N, window, 1).
    y_test : np.ndarray
        Test targets. Shape: (N, 1).

    Returns
    -------
    dict
        Dictionary with metrics:
        {"mae": float, "rmse": float}
    """
    y_pred = model.predict(X_test, verbose=0)
    return {"mae": mae(y_test, y_pred), "rmse": rmse(y_test, y_pred)}


# ----------------------------
# Visualization / demo (provided)
# ----------------------------

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, k: int = 250) -> None:
    """Plot the first k points of true vs predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True targets. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predictions. Shape: (N,) or (N, 1).
    k : int
        Number of points to plot.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    k = int(min(k, len(y_true), len(y_pred)))

    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:k], label="true")
    plt.plot(y_pred[:k], label="pred", alpha=0.9)
    plt.grid(True)
    plt.legend()
    plt.title("One-step forecast (first k test points)")
    plt.show()


def demo() -> None:
    """Run a small demo on synthetic data (for orientation, NOT used in tests).

    The demo uses a fixed seed and a known signal (trend + seasonality + noise).
    Students may use it to validate that their pipeline works end-to-end.
    """
    # Make demo deterministic
    tf.keras.utils.set_random_seed(123)
    rng = np.random.default_rng(123)

    t = np.arange(1200, dtype=np.float32)
    series = (
        0.001 * t
        + 2.0 * np.sin(2 * np.pi * t / 50.0)
        + 0.8 * np.sin(2 * np.pi * t / 16.0)
        + rng.normal(0, 0.2, size=len(t)).astype(np.float32)
    )

    model, X_test, y_test, _ = train_model(series, window=40, epochs=20, batch_size=64, seed=123, verbose=0)
    metrics = evaluate_model(model, X_test, y_test)
    print("Test metrics:", metrics)

    y_pred = model.predict(X_test, verbose=0)
    plot_predictions(y_test, y_pred, k=250)


if __name__ == "__main__":
    demo()
