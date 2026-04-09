"""
Feature Engineering for AML Detection
Produces three feature families:
  - Behavioral  : amount stats, frequency, time patterns per account
  - Graph       : degree, betweenness, PageRank, clustering coefficient
  - Temporal    : velocity, burst detection, night-time activity
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── behavioral features ───────────────────────────────────────────────────────

def _behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features derived from sender / receiver history."""
    feats = df[["txn_id", "sender", "receiver", "amount", "timestamp", "country"]].copy()

    # Rolling sender stats (computed over entire history — fine for a portfolio demo)
    sender_stats = (
        df.groupby("sender")["amount"]
        .agg(sender_txn_count="count",
             sender_mean_amt="mean",
             sender_std_amt="std",
             sender_max_amt="max",
             sender_total_vol="sum")
        .reset_index()
        .fillna(0)
    )
    feats = feats.merge(sender_stats, on="sender", how="left")

    # Receiver in-degree stats
    receiver_stats = (
        df.groupby("receiver")["amount"]
        .agg(recv_txn_count="count",
             recv_mean_amt="mean",
             recv_total_vol="sum")
        .reset_index()
        .rename(columns={"receiver": "recv_id"})
    )
    feats = feats.merge(receiver_stats, left_on="receiver", right_on="recv_id", how="left").drop("recv_id", axis=1)

    # Amount normalised by sender history
    feats["amt_vs_sender_mean"] = feats["amount"] / (feats["sender_mean_amt"] + 1e-9)

    # High-risk country flag
    high_risk = {"PA", "KY", "VG", "CY", "LB"}
    feats["high_risk_country"] = feats["country"].isin(high_risk).astype(int)

    # Just-under-threshold flag ($9,000–$10,000 band)
    feats["near_ctr_threshold"] = ((feats["amount"] >= 9000) & (feats["amount"] < 10000)).astype(int)

    # Hour of day
    feats["hour"] = pd.to_datetime(feats["timestamp"]).dt.hour
    feats["is_night"] = ((feats["hour"] >= 22) | (feats["hour"] <= 5)).astype(int)

    return feats.drop(["sender", "receiver", "country", "timestamp"], axis=1)


# ── graph features ────────────────────────────────────────────────────────────

def _graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a directed transaction graph and compute per-node centrality metrics.
    Returns a DataFrame indexed by account ID.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        s, r, a = row["sender"], row["receiver"], row["amount"]
        if G.has_edge(s, r):
            G[s][r]["weight"] += a
            G[s][r]["count"]  += 1
        else:
            G.add_edge(s, r, weight=a, count=1)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Degree (raw activity signal)
    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    # PageRank — laundering accounts are often high-centrality hubs
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200)

    # Strongly connected components (useful for round-tripping)
    scc = nx.strongly_connected_components(G)
    scc_map = {}
    for component in scc:
        size = len(component)
        for node in component:
            scc_map[node] = size

    # Clustering on undirected version (dense local neighbourhoods = smurfing)
    G_undirected = G.to_undirected()
    clustering   = nx.clustering(G_undirected)

    nodes = list(G.nodes())
    graph_df = pd.DataFrame({
        "account":          nodes,
        "in_degree":        [in_degree.get(n, 0)  for n in nodes],
        "out_degree":       [out_degree.get(n, 0) for n in nodes],
        "pagerank":         [pagerank.get(n, 0)   for n in nodes],
        "scc_size":         [scc_map.get(n, 1)    for n in nodes],
        "local_clustering": [clustering.get(n, 0) for n in nodes],
    })
    return graph_df


# ── velocity features ─────────────────────────────────────────────────────────

def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count transactions per sender in 1-hour and 24-hour windows."""
    df = df.sort_values("timestamp").copy()

    # For each transaction, count how many other txns the sender had in ±1h
    sender_times = df.groupby("sender")["timestamp"].apply(list).to_dict()

    velocities_1h  = []
    velocities_24h = []

    for _, row in df.iterrows():
        times = sender_times.get(row["sender"], [])
        t0 = row["timestamp"]
        cnt_1h  = sum(1 for t in times if abs((t - t0).total_seconds()) <= 3600)
        cnt_24h = sum(1 for t in times if abs((t - t0).total_seconds()) <= 86400)
        velocities_1h.append(cnt_1h)
        velocities_24h.append(cnt_24h)

    df = df.copy()
    df["velocity_1h"]  = velocities_1h
    df["velocity_24h"] = velocities_24h
    return df[["txn_id", "velocity_1h", "velocity_24h"]]


# ── master feature builder ────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "amount", "sender_txn_count", "sender_mean_amt", "sender_std_amt",
    "sender_max_amt", "sender_total_vol", "recv_txn_count", "recv_mean_amt",
    "recv_total_vol", "amt_vs_sender_mean", "high_risk_country",
    "near_ctr_threshold", "hour", "is_night",
    "in_degree", "out_degree", "pagerank", "scc_size", "local_clustering",
    "velocity_1h", "velocity_24h",
]


def build_features(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Returns:
        X_scaled  : np.ndarray  — scaled numeric features
        y         : np.ndarray  — binary labels
        feature_df: pd.DataFrame — unscaled features (for inspection)
        scaler    : StandardScaler
        graph_df  : pd.DataFrame — per-account graph metrics
    """
    if verbose:
        print("Building behavioral features...")
    behav = _behavioral_features(df)

    if verbose:
        print("Building graph features...")
    graph_df = _graph_features(df)

    # Map graph features back to transactions via sender
    sender_graph = graph_df.rename(columns={
        "account": "sender",
        "in_degree": "in_degree", "out_degree": "out_degree",
        "pagerank": "pagerank", "scc_size": "scc_size",
        "local_clustering": "local_clustering",
    })
    merged = df[["txn_id", "sender", "is_laundering"]].merge(
        behav, on="txn_id", how="left"
    ).merge(
        sender_graph, on="sender", how="left"
    ).drop("sender", axis=1).fillna(0)

    if verbose:
        print("Building velocity features...")
    vel = _velocity_features(df)
    merged = merged.merge(vel, on="txn_id", how="left").fillna(0)

    y = merged["is_laundering"].values
    feature_df = merged.drop(["txn_id", "is_laundering"], axis=1)

    # Keep only numeric columns
    feature_df = feature_df[NUMERIC_FEATURES].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    if verbose:
        print(f"Feature matrix: {X_scaled.shape} | Fraud rate: {y.mean():.2%}")

    return X_scaled, y, feature_df, scaler, graph_df
