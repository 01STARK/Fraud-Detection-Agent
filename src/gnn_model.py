"""
Graph Neural Network — message-passing implementation with NumPy + NetworkX.
No PyTorch Geometric required.

Two rounds of neighbourhood aggregation (mean pooling), followed by
a 2-layer MLP classifier.  Training uses binary cross-entropy + SGD.

This demonstrates the GNN concept explicitly:
  h_v^(k) = ReLU( W_k · MEAN({ h_u^(k-1) : u ∈ N(v) ∪ {v} }) )
"""

import numpy as np
import networkx as nx
import pandas as pd


# ── activation helpers ────────────────────────────────────────────────────────

def relu(x):         return np.maximum(0, x)
def relu_grad(x):    return (x > 0).astype(float)
def sigmoid(x):      return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def bce(y, p):       return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

def _he(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


# ── graph construction ────────────────────────────────────────────────────────

def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Each account is a node.
    Edge weight = total flow between sender → receiver.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        s, r, a = row["sender"], row["receiver"], float(row["amount"])
        if G.has_edge(s, r):
            G[s][r]["weight"] += a
        else:
            G.add_edge(s, r, weight=a)
    return G


def graph_node_features(G: nx.DiGraph, graph_df: pd.DataFrame) -> tuple:
    """
    Build node feature matrix H from graph_df (in/out-degree, pagerank, …).
    Returns (nodes_list, H_matrix).
    """
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    feat_cols = ["in_degree", "out_degree", "pagerank", "scc_size", "local_clustering"]
    lookup = graph_df.set_index("account")[feat_cols].to_dict("index")

    H = np.zeros((len(nodes), len(feat_cols)), dtype=float)
    for node in nodes:
        i = node_idx[node]
        if node in lookup:
            for j, col in enumerate(feat_cols):
                H[i, j] = lookup[node][col]

    # Normalise columns
    col_std = H.std(axis=0) + 1e-9
    H = (H - H.mean(axis=0)) / col_std

    return nodes, node_idx, H


# ── message passing ───────────────────────────────────────────────────────────

def message_passing(G: nx.DiGraph, H: np.ndarray,
                    node_idx: dict, n_rounds: int = 2) -> np.ndarray:
    """
    Aggregate neighbourhood representations (mean pooling).
    Uses UNDIRECTED neighbourhood so information flows both ways.
    Returns aggregated node embeddings of shape (N, D).
    """
    G_undir = G.to_undirected()
    N, D = H.shape
    nodes = list(node_idx.keys())

    for _ in range(n_rounds):
        H_new = np.zeros_like(H)
        for node in nodes:
            i = node_idx[node]
            neighbours = list(G_undir.neighbors(node)) + [node]  # self-loop
            nb_indices  = [node_idx[nb] for nb in neighbours if nb in node_idx]
            if nb_indices:
                H_new[i] = H[nb_indices].mean(axis=0)
            else:
                H_new[i] = H[i]
        H = H_new

    return H


# ── GNN classifier ────────────────────────────────────────────────────────────

class GNNClassifier:
    """
    2-round message-passing GNN with a 2-layer MLP head.
    Trained at the NODE level (each account has a binary label = 1 if
    it participated in any flagged transaction).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 lr: float = 0.01, epochs: int = 150, random_state: int = 42):
        self.rng        = np.random.default_rng(random_state)
        self.epochs     = epochs
        self.lr         = lr
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.train_losses: list = []

        # MLP weights: (input_dim → hidden_dim → 1)
        self.W1 = _he(input_dim, hidden_dim, self.rng)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = _he(hidden_dim, 1, self.rng)
        self.b2 = np.zeros((1, 1))

    # ── forward ──────────────────────────────────────────────────────────────

    def _forward(self, H: np.ndarray):
        z1 = H @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        out = sigmoid(z2).ravel()
        return out, z1, a1

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, H: np.ndarray, y: np.ndarray, verbose: bool = True) -> "GNNClassifier":
        """H: (N, D) graph embeddings; y: (N,) binary node labels."""
        N = H.shape[0]
        # Class weighting to handle imbalance
        pos = y.sum()
        neg = N - pos
        w_pos = neg / (pos + 1e-9)
        sample_weights = np.where(y == 1, w_pos, 1.0)

        print(f"  GNN training: {N} nodes, {int(pos)} positive, {self.epochs} epochs")

        for epoch in range(self.epochs):
            probs, z1, a1 = self._forward(H)

            # Weighted BCE gradient
            delta2 = ((probs - y) * sample_weights / N).reshape(-1, 1)
            dW2 = a1.T @ delta2
            db2 = delta2.sum(axis=0, keepdims=True)

            delta1 = (delta2 @ self.W2.T) * relu_grad(z1)
            dW1 = H.T @ delta1
            db1 = delta1.sum(axis=0, keepdims=True)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            loss = bce(y, probs)
            self.train_losses.append(loss)

            if verbose and (epoch + 1) % 30 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs} — BCE loss: {loss:.4f}")

        return self

    def predict_proba(self, H: np.ndarray) -> np.ndarray:
        probs, _, _ = self._forward(H)
        return probs

    def tune_threshold(self, H: np.ndarray, y: np.ndarray,
                       beta: float = 2.0) -> dict:
        from sklearn.metrics import (
            precision_recall_curve, roc_auc_score,
            average_precision_score,
        )
        scores = self.predict_proba(H)
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


# ── node-level labels ─────────────────────────────────────────────────────────

def build_node_labels(df: pd.DataFrame, nodes: list) -> np.ndarray:
    """
    A node (account) is labelled 1 if it appeared in ANY laundering transaction.
    """
    bad_senders   = set(df[df["is_laundering"] == 1]["sender"])
    bad_receivers = set(df[df["is_laundering"] == 1]["receiver"])
    bad_accounts  = bad_senders | bad_receivers
    return np.array([1 if n in bad_accounts else 0 for n in nodes])
