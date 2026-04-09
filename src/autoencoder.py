"""
Autoencoder for Anomaly Detection — pure NumPy implementation.
Architecture: input → 16 → 8 → 4 → 8 → 16 → input
Anomaly score = mean reconstruction error per sample.
High reconstruction error ≈ the encoder never saw this pattern → suspicious.
"""

import numpy as np


# ── activation functions ──────────────────────────────────────────────────────

def relu(x):      return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)
def sigmoid(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── weight initialisation ─────────────────────────────────────────────────────

def _he_init(fan_in: int, fan_out: int, rng) -> np.ndarray:
    """He initialisation — good for ReLU networks."""
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


# ── autoencoder class ─────────────────────────────────────────────────────────

class NumpyAutoencoder:
    """
    Symmetric autoencoder with ReLU hidden layers and linear output.
    Trained with mini-batch gradient descent + momentum.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple = (16, 8, 4),
                 lr: float = 0.003, momentum: float = 0.9,
                 batch_size: int = 256, epochs: int = 60,
                 random_state: int = 42):
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.lr          = lr
        self.momentum    = momentum
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.rng         = np.random.default_rng(random_state)

        # Build symmetric layer sizes: [input, *hidden, *hidden[::-1], input]
        enc = [input_dim] + list(hidden_dims)
        dec = list(hidden_dims[-2::-1]) + [input_dim]
        self.layer_sizes = enc + dec

        self._init_weights()
        self.train_losses: list = []

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        self.W, self.b   = [], []
        self.vW, self.vb = [], []     # momentum buffers
        for i in range(len(self.layer_sizes) - 1):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            self.W.append(_he_init(fan_in, fan_out, self.rng))
            self.b.append(np.zeros((1, fan_out)))
            self.vW.append(np.zeros((fan_in, fan_out)))
            self.vb.append(np.zeros((1, fan_out)))

    # ── forward pass ─────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> tuple:
        """Returns output and list of (pre-activation, post-activation) per layer."""
        activations = []
        a = X
        n_layers = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            # Last layer: linear (reconstruction); all others: ReLU
            a_new = z if i == n_layers - 1 else relu(z)
            activations.append((z, a_new))
            a = a_new
        return a, activations

    # ── backward pass ────────────────────────────────────────────────────────

    def _backward(self, X: np.ndarray, activations: list) -> tuple:
        """MSE loss backprop. Returns gradient lists (dW, db)."""
        m      = X.shape[0]
        output = activations[-1][1]
        delta  = (output - X) / m          # d MSE / d output (linear output layer)

        dWs = [None] * len(self.W)
        dbs = [None] * len(self.b)

        a_prev = X
        for i in reversed(range(len(self.W))):
            z, a = activations[i]
            a_prev_i = activations[i - 1][1] if i > 0 else X

            if i < len(self.W) - 1:       # ReLU gradient
                delta = delta * relu_grad(z)

            dWs[i] = a_prev_i.T @ delta
            dbs[i] = delta.sum(axis=0, keepdims=True)

            delta = delta @ self.W[i].T

        return dWs, dbs

    # ── weight update ─────────────────────────────────────────────────────────

    def _update(self, dWs, dbs):
        for i in range(len(self.W)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dWs[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * dbs[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, verbose: bool = True) -> "NumpyAutoencoder":
        """Train only on LEGITIMATE transactions to learn the normal pattern."""
        n = X.shape[0]
        print(f"  Training autoencoder on {n} samples, "
              f"{self.epochs} epochs, lr={self.lr}")

        for epoch in range(self.epochs):
            # Shuffle
            idx = self.rng.permutation(n)
            X_shuffled = X[idx]
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch = X_shuffled[start:start + self.batch_size]
                out, acts = self._forward(batch)
                loss = float(np.mean((out - batch) ** 2))
                epoch_loss += loss * len(batch)
                dWs, dbs = self._backward(batch, acts)
                self._update(dWs, dbs)

            avg_loss = epoch_loss / n
            self.train_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs} — MSE loss: {avg_loss:.6f}")

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        out, _ = self._forward(X)
        return out

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample mean squared reconstruction error.
        High error = pattern was NOT learned by encoder = suspicious.
        Normalised using stored min/max so single-sample scoring stays calibrated.
        """
        out  = self.reconstruct(X)
        mse  = np.mean((X - out) ** 2, axis=1)
        lo   = getattr(self, '_score_min', mse.min())
        hi   = getattr(self, '_score_max', mse.max())
        return (mse - lo) / (hi - lo + 1e-9)

    def tune_threshold(self, X: np.ndarray, y: np.ndarray,
                       beta: float = 2.0) -> dict:
        from sklearn.metrics import (
            precision_recall_curve, roc_auc_score,
            average_precision_score, classification_report,
        )
        # Store MSE range for consistent single-sample inference
        out  = self.reconstruct(X)
        mse  = np.mean((X - out) ** 2, axis=1)
        self._score_min = float(mse.min())
        self._score_max = float(mse.max())
        scores = self.anomaly_scores(X)
        precisions, recalls, thresholds = precision_recall_curve(y, scores)
        f_beta = ((1 + beta**2) * precisions * recalls /
                  (beta**2 * precisions + recalls + 1e-9))
        best_idx = np.argmax(f_beta[:-1])
        self.threshold_ = float(thresholds[best_idx])

        y_pred = (scores >= self.threshold_).astype(int)

        return {
            "threshold":     self.threshold_,
            "precision":     float(precisions[best_idx]),
            "recall":        float(recalls[best_idx]),
            "f_beta":        float(f_beta[best_idx]),
            "roc_auc":       float(roc_auc_score(y, scores)),
            "avg_precision": float(average_precision_score(y, scores)),
            "precisions":    precisions,
            "recalls":       recalls,
            "thresholds":    thresholds,
            "f_betas":       f_beta,
            "scores":        scores,
            "y_pred":        y_pred,
        }