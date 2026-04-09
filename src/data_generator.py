"""
Synthetic AML Transaction Data Generator
Embeds four real-world money laundering patterns:
  1. Smurfing      — many small transactions just under reporting threshold
  2. Layering      — rapid multi-hop fund movement through accounts
  3. Round-tripping — funds leave and return to the same account
  4. Structuring   — repeated deposits just below $10k CTR threshold
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
RNG  = np.random.default_rng(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────

def _rand_account(prefix: str, n: int) -> np.ndarray:
    return np.array([f"{prefix}{RNG.integers(1000, 9999)}" for _ in range(n)])

def _rand_country(n: int, high_risk_prob: float = 0.1) -> np.ndarray:
    normal    = ["US", "GB", "DE", "CA", "AU", "FR", "JP"]
    high_risk = ["PA", "KY", "VG", "CY", "LB"]          # common shell-corp jurisdictions
    countries = []
    for _ in range(n):
        if RNG.random() < high_risk_prob:
            countries.append(RNG.choice(high_risk))
        else:
            countries.append(RNG.choice(normal))
    return np.array(countries)

# ── legitimate transactions ───────────────────────────────────────────────────

def _generate_legit(n: int = 8000) -> pd.DataFrame:
    base_time = datetime(2024, 1, 1)
    seconds   = RNG.integers(0, 365 * 86400, n)
    amounts   = np.clip(RNG.lognormal(mean=5.5, sigma=1.2, size=n), 1, 50_000)
    senders   = _rand_account("ACCT", n)
    receivers = _rand_account("ACCT", n)
    categories = RNG.choice(
        ["retail", "food", "travel", "utilities", "healthcare",
         "entertainment", "transfer", "payroll"],
        size=n, p=[0.25, 0.15, 0.10, 0.10, 0.08, 0.07, 0.20, 0.05]
    )
    return pd.DataFrame({
        "txn_id":    [f"TXN{i:06d}" for i in range(n)],
        "timestamp": [base_time + timedelta(seconds=int(s)) for s in seconds],
        "sender":    senders,
        "receiver":  receivers,
        "amount":    np.round(amounts, 2),
        "category":  categories,
        "country":   _rand_country(n, high_risk_prob=0.05),
        "is_laundering": 0,
        "pattern":   "legit",
    })

# ── pattern 1: smurfing ───────────────────────────────────────────────────────

def _generate_smurfing(n_groups: int = 30) -> pd.DataFrame:
    rows = []
    txn_counter = 90_000
    base_time = datetime(2024, 1, 1)
    for g in range(n_groups):
        n_txns    = RNG.integers(8, 25)          # many small transactions
        master    = f"SMURF_MASTER_{g:03d}"
        mules     = [f"MULE_{g}_{m}" for m in range(n_txns)]
        t0        = base_time + timedelta(days=int(RNG.integers(0, 360)))
        for i, mule in enumerate(mules):
            amt = round(float(RNG.uniform(800, 2500)), 2)  # under threshold
            rows.append({
                "txn_id":    f"TXN{txn_counter:06d}",
                "timestamp": t0 + timedelta(minutes=int(i * RNG.integers(5, 30))),
                "sender":    mule,
                "receiver":  master,
                "amount":    amt,
                "category":  "transfer",
                "country":   RNG.choice(["US", "PA", "KY"]),
                "is_laundering": 1,
                "pattern":   "smurfing",
            })
            txn_counter += 1
    return pd.DataFrame(rows)

# ── pattern 2: layering ───────────────────────────────────────────────────────

def _generate_layering(n_chains: int = 25) -> pd.DataFrame:
    rows = []
    txn_counter = 80_000
    base_time = datetime(2024, 1, 1)
    for c in range(n_chains):
        hops   = RNG.integers(4, 9)
        chain  = [f"LAYER_{c}_{h}" for h in range(hops + 1)]
        amount = round(float(RNG.uniform(20_000, 200_000)), 2)
        t0     = base_time + timedelta(days=int(RNG.integers(0, 360)))
        for i in range(hops):
            # Each hop slightly reduces the amount (fees / splitting)
            hop_amt = round(amount * float(RNG.uniform(0.85, 0.98)), 2)
            rows.append({
                "txn_id":    f"TXN{txn_counter:06d}",
                "timestamp": t0 + timedelta(hours=int(i * RNG.integers(1, 12))),
                "sender":    chain[i],
                "receiver":  chain[i + 1],
                "amount":    hop_amt,
                "category":  "transfer",
                "country":   RNG.choice(["LB", "CY", "VG", "PA", "US"]),
                "is_laundering": 1,
                "pattern":   "layering",
            })
            txn_counter += 1
            amount = hop_amt
    return pd.DataFrame(rows)

# ── pattern 3: round-tripping ─────────────────────────────────────────────────

def _generate_round_trip(n: int = 40) -> pd.DataFrame:
    rows = []
    txn_counter = 70_000
    base_time = datetime(2024, 1, 1)
    for i in range(n):
        origin = f"ORIGIN_{i:03d}"
        shell  = f"SHELL_{i:03d}"
        amount = round(float(RNG.uniform(50_000, 500_000)), 2)
        t0     = base_time + timedelta(days=int(RNG.integers(0, 300)))
        # Out leg
        rows.append({
            "txn_id": f"TXN{txn_counter:06d}",
            "timestamp": t0,
            "sender": origin, "receiver": shell,
            "amount": amount, "category": "investment",
            "country": RNG.choice(["KY", "VG", "PA"]),
            "is_laundering": 1, "pattern": "round_trip",
        })
        txn_counter += 1
        # Return leg (slightly less — laundered "profit")
        rows.append({
            "txn_id": f"TXN{txn_counter:06d}",
            "timestamp": t0 + timedelta(days=int(RNG.integers(30, 90))),
            "sender": shell, "receiver": origin,
            "amount": round(amount * float(RNG.uniform(1.05, 1.20)), 2),
            "category": "investment_return",
            "country": RNG.choice(["KY", "VG", "PA"]),
            "is_laundering": 1, "pattern": "round_trip",
        })
        txn_counter += 1
    return pd.DataFrame(rows)

# ── pattern 4: structuring ────────────────────────────────────────────────────

def _generate_structuring(n_actors: int = 35) -> pd.DataFrame:
    rows = []
    txn_counter = 60_000
    base_time = datetime(2024, 1, 1)
    CTR_THRESHOLD = 10_000
    for a in range(n_actors):
        actor  = f"STRUCT_{a:03d}"
        n_deps = RNG.integers(5, 15)
        t0     = base_time + timedelta(days=int(RNG.integers(0, 340)))
        for d in range(n_deps):
            amt = round(float(RNG.uniform(CTR_THRESHOLD * 0.88, CTR_THRESHOLD * 0.99)), 2)
            rows.append({
                "txn_id":    f"TXN{txn_counter:06d}",
                "timestamp": t0 + timedelta(days=int(d * RNG.integers(1, 5))),
                "sender":    actor,
                "receiver":  f"BANK_{RNG.integers(1, 10):02d}",
                "amount":    amt,
                "category":  "deposit",
                "country":   "US",
                "is_laundering": 1,
                "pattern":   "structuring",
            })
            txn_counter += 1
    return pd.DataFrame(rows)

# ── public API ────────────────────────────────────────────────────────────────

def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Return a shuffled DataFrame combining all patterns."""
    legit       = _generate_legit(8000)
    smurfing    = _generate_smurfing(30)
    layering    = _generate_layering(25)
    round_trip  = _generate_round_trip(40)
    structuring = _generate_structuring(35)

    df = pd.concat([legit, smurfing, layering, round_trip, structuring], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df["pattern"].value_counts())
    print(f"\nFraud rate: {df['is_laundering'].mean():.2%}")
