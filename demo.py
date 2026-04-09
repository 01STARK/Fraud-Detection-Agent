"""
Demo — AI Fraud Detection Agent
================================
Runs three sample transactions through the full 4-step pipeline:
  1. Clean retail payment            → expected: APPROVE
  2. Near-threshold structuring txn  → expected: FLAG / MONITOR
  3. High-value offshore transfer    → expected: BLOCK / FLAG
"""

from datetime import datetime
from agent import FraudDetectionAgent


SAMPLE_TRANSACTIONS = [
    # ── 1. Normal retail payment ──────────────────────────────────────────────
    {
        "txn_id": "TXN_DEMO_001",
        "sender": "ACCT_ALICE_001",
        "receiver": "ACCT_SHOPMART",
        "amount": 87.45,
        "timestamp": datetime(2025, 3, 15, 14, 22, 0),
        "country": "US",
        "category": "retail",
    },

    # ── 2. Structuring — just under CTR threshold ─────────────────────────────
    {
        "txn_id": "TXN_DEMO_002",
        "sender": "STRUCT_ACCT_777",
        "receiver": "BANK_03",
        "amount": 9_450.00,
        "timestamp": datetime(2025, 3, 20, 2, 45, 0),   # 2:45 AM
        "country": "US",
        "category": "deposit",
    },

    # ── 3. Large offshore wire — high-risk jurisdiction ───────────────────────
    {
        "txn_id": "TXN_DEMO_003",
        "sender": "ACCT_OFFSHORE_X",
        "receiver": "ACCT_SHELL_PA",
        "amount": 48_000.00,
        "timestamp": datetime(2025, 3, 22, 23, 55, 0),  # 11:55 PM
        "country": "PA",                                  # Panama
        "category": "transfer",
    },
]


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AI FRAUD DETECTION AGENT — DEMO")
    print("  3 transactions  ·  4 steps each")
    print("=" * 65)

    agent = FraudDetectionAgent(verbose=True)

    results = []
    for txn in SAMPLE_TRANSACTIONS:
        report = agent.run(txn)
        results.append({
            "txn_id": report["transaction"]["id"],
            "amount": report["transaction"]["amount"],
            "ml_score": report["ml_score"],
            "risk_level": report["risk_level"],
            "action": report["decision"]["action"],
        })

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'TXN ID':<20} {'AMOUNT':>12}  {'SCORE':>6}  {'RISK':<9}  ACTION")
    print("  " + "-" * 61)
    for r in results:
        print(
            f"  {r['txn_id']:<20} ${r['amount']:>11,.2f}  {r['ml_score']:.4f}"
            f"  {r['risk_level']:<9}  {r['action']}"
        )
    print("=" * 65)
    print("\n  Reports saved to: reports/\n")
