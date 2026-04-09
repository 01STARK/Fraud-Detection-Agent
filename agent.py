"""
AI Fraud Detection Agent System
================================
Step 1 — ML scoring      : IsolationForest (AML project) flags anomalies
Step 2 — LLM explanation : Groq/LLaMA explains which features drove the score
Step 3 — Agent decision  : Groq tool-use picks BLOCK / FLAG / MONITOR / APPROVE
Step 4 — Report          : Structured JSON + human-readable summary saved to disk

LLM: Groq (FREE — llama-3.3-70b-versatile)
Get free key: console.groq.com  (no credit card required)
"""

import sys
import os
import json
import textwrap
from datetime import datetime
from pathlib import Path

# Load .env manually
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import pandas as pd
from groq import Groq

# ── load AML source code ──────────────────────────────────────────────────────
AML_SRC = Path(__file__).parent.parent / "08 AML" / "aml_detection"
sys.path.insert(0, str(AML_SRC))

from src.data_generator import generate_dataset
from src.isolation_forest_model import AMLIsolationForest
from src.feature_engineering import build_features, NUMERIC_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RISK_THRESHOLDS = {"BLOCK": 0.75, "FLAG": 0.50, "MONITOR": 0.25}

# ── agent decision tools (OpenAI-compatible format) ───────────────────────────
DECISION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "block_transaction",
            "description": (
                "Immediately block the transaction. Use when risk score ≥ 0.75 "
                "or multiple strong fraud signals are present."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason":      {"type": "string", "description": "Why the transaction is blocked."},
                    "risk_score":  {"type": "number"},
                    "key_signals": {"type": "array", "items": {"type": "string"}, "description": "Top 3 fraud signals."},
                },
                "required": ["reason", "risk_score", "key_signals"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_review",
            "description": "Flag for human analyst review. Use for moderate risk (0.50–0.74) or ambiguous signals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason":        {"type": "string"},
                    "risk_score":    {"type": "number"},
                    "analyst_notes": {"type": "string", "description": "What the analyst should investigate."},
                    "key_signals":   {"type": "array", "items": {"type": "string"}},
                },
                "required": ["reason", "risk_score", "analyst_notes", "key_signals"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "monitor_account",
            "description": "Allow transaction but place sender under enhanced monitoring. Use for low-moderate risk (0.25–0.49).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason":        {"type": "string"},
                    "risk_score":    {"type": "number"},
                    "duration_days": {"type": "integer", "description": "Days to monitor: 7, 14, or 30."},
                    "key_signals":   {"type": "array", "items": {"type": "string"}},
                },
                "required": ["reason", "risk_score", "duration_days", "key_signals"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_transaction",
            "description": "Approve as legitimate. Use when risk score < 0.25 and no fraud signals present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason":     {"type": "string"},
                    "risk_score": {"type": "number"},
                },
                "required": ["reason", "risk_score"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
class FraudDetectionAgent:
    """
    Four-step AI fraud detection pipeline:
      1. ML anomaly score   (IsolationForest)
      2. LLM explanation    (Groq / LLaMA-3.3-70B)
      3. Agent decision     (Groq tool use — BLOCK / FLAG / MONITOR / APPROVE)
      4. Report generation  (JSON + text saved to disk)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com "
                "and add GROQ_API_KEY=... to your .env file."
            )
        self._client = Groq(api_key=api_key)
        self._train_model()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── model training ────────────────────────────────────────────────────────

    def _train_model(self) -> None:
        self._log("\n[INIT] Generating synthetic AML training data…")
        df = generate_dataset()
        legit = df[df["is_laundering"] == 0].sample(1200, random_state=42)
        fraud = df[df["is_laundering"] == 1]
        self._context_df = pd.concat([legit, fraud]).reset_index(drop=True)

        self._log("[INIT] Building features…")
        X, y, feat_df, scaler, _ = build_features(self._context_df, verbose=False)
        self._scaler = scaler
        self._feature_cols = NUMERIC_FEATURES

        self._log("[INIT] Training IsolationForest…")
        self._model = AMLIsolationForest(contamination=0.08)
        self._model.fit(X)
        metrics = self._model.tune_threshold(X, y, beta=2.0)
        self._log(
            f"[INIT] Model ready — AUC={metrics['roc_auc']:.3f}  "
            f"Recall={metrics['recall']:.3f}  Threshold={metrics['threshold']:.3f}"
        )

    # ── step 1 : ML scoring ───────────────────────────────────────────────────

    def _ml_score(self, txn: dict) -> tuple[float, dict]:
        row = {
            "txn_id":    txn.get("txn_id", "TXN_NEW"),
            "timestamp": pd.to_datetime(txn.get("timestamp", datetime.now())),
            "sender":    txn["sender"],
            "receiver":  txn["receiver"],
            "amount":    float(txn["amount"]),
            "category":  txn.get("category", "transfer"),
            "country":   txn.get("country", "US"),
            "is_laundering": 0,
            "pattern":   "unknown",
        }
        augmented = pd.concat(
            [self._context_df, pd.DataFrame([row])], ignore_index=True
        )
        _, _, feat_df, _, _ = build_features(augmented, verbose=False)
        new_feat = feat_df.iloc[[-1]][self._feature_cols].fillna(0)
        X_scaled = self._scaler.transform(new_feat)
        score = float(self._model.anomaly_scores(X_scaled)[0])
        return score, new_feat.iloc[0].to_dict()

    # ── step 2 : LLM explanation ──────────────────────────────────────────────

    def _explain_prompt(self, txn: dict, score: float, features: dict) -> str:
        top = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        feat_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in top)
        return textwrap.dedent(f"""
            You are an AML (Anti-Money Laundering) analyst assistant.

            A transaction has been scored by an IsolationForest model.
            Anomaly score: {score:.4f}  (0 = normal, 1 = highly anomalous)

            Transaction details:
            - ID: {txn.get('txn_id', 'N/A')}
            - Amount: ${txn['amount']:,.2f}
            - Sender: {txn['sender']}
            - Receiver: {txn['receiver']}
            - Country: {txn.get('country', 'US')}
            - Category: {txn.get('category', 'transfer')}
            - Timestamp: {txn.get('timestamp', 'N/A')}

            Top contributing features:
            {feat_lines}

            Explain in 3–5 sentences:
            1. Which features are most anomalous and why.
            2. What laundering pattern this might resemble (smurfing, layering, structuring, round-tripping).
            3. Any factors that increase or decrease concern.

            Be concise and factual. Do not make definitive conclusions.
        """).strip()

    def stream_explanation(self, txn: dict, score: float, features: dict):
        """Yields text chunks for Streamlit st.write_stream()."""
        prompt = self._explain_prompt(txn, score, features)
        stream = self._client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text

    def _llm_explain(self, txn: dict, score: float, features: dict) -> str:
        self._log("\n[STEP 2] Requesting anomaly explanation from Groq…")
        prompt = self._explain_prompt(txn, score, features)
        resp = self._client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    # ── step 3 : agent decision ───────────────────────────────────────────────

    def _agent_decide(self, txn: dict, score: float, explanation: str) -> dict:
        prompt = textwrap.dedent(f"""
            You are an autonomous fraud decision agent.

            Transaction:
            - ID: {txn.get('txn_id', 'N/A')}
            - Amount: ${txn['amount']:,.2f}
            - Route: {txn['sender']} → {txn['receiver']}
            - Country: {txn.get('country', 'US')}
            - ML Anomaly Score: {score:.4f}  (0=normal, 1=highly anomalous)

            Analyst Explanation:
            {explanation}

            Call EXACTLY ONE tool based on the risk level.
        """).strip()

        self._log("\n[STEP 3] Agent deciding action…")
        resp = self._client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=DECISION_TOOLS,
            tool_choice="required",
            max_tokens=512,
        )

        action_map = {
            "block_transaction":   "BLOCK",
            "flag_for_review":     "FLAG_FOR_REVIEW",
            "monitor_account":     "MONITOR",
            "approve_transaction": "APPROVE",
        }
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc   = msg.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            return {"action": action_map.get(name, name.upper()), **args}

        return {"action": "FLAG_FOR_REVIEW", "reason": "No tool call returned.", "risk_score": score}

    # ── step 4 : report ───────────────────────────────────────────────────────

    def _generate_report(self, txn, score, features, explanation, decision) -> dict:
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "transaction": {
                "id":       txn.get("txn_id", "N/A"),
                "amount":   txn["amount"],
                "sender":   txn["sender"],
                "receiver": txn["receiver"],
                "country":  txn.get("country", "US"),
                "category": txn.get("category", "transfer"),
                "timestamp": str(txn.get("timestamp", "N/A")),
            },
            "ml_score":    round(score, 4),
            "risk_level":  self._risk_level(score),
            "key_features": {
                k: round(v, 4)
                for k, v in sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            },
            "explanation": explanation,
            "decision":    decision,
        }

        icons = {"BLOCK": "🔴", "FLAG_FOR_REVIEW": "🟡", "MONITOR": "🟠", "APPROVE": "🟢"}
        action = decision["action"]
        div = "=" * 65
        print(f"\n{div}")
        print(f"  FRAUD DETECTION REPORT  —  {txn.get('txn_id', 'N/A')}")
        print(div)
        print(f"  Amount   : ${txn['amount']:>12,.2f}")
        print(f"  Route    : {txn['sender']} → {txn['receiver']}")
        print(f"  Country  : {txn.get('country', 'US')}")
        print(f"  ML Score : {score:.4f}  [{self._risk_level(score)}]")
        print(f"\n  DECISION : {icons.get(action,'⚪')} {action}")
        print(f"  Reason   : {decision.get('reason', '')}")
        if "key_signals"    in decision:
            for s in decision["key_signals"]: print(f"             • {s}")
        if "analyst_notes"  in decision: print(f"  Notes    : {decision['analyst_notes']}")
        if "duration_days"  in decision: print(f"  Monitor  : {decision['duration_days']} days")
        print(f"\n  Explanation:\n")
        for line in textwrap.wrap(explanation, 60): print(f"    {line}")
        print(div)

        txn_id = txn.get("txn_id", "TXN").replace(" ", "_")
        path = REPORTS_DIR / f"{txn_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2))
        self._log(f"\n[REPORT] Saved → {path}")
        return report

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, transaction: dict) -> dict:
        txn_id = transaction.get("txn_id", "?")
        self._log(f"\n{'─'*65}\n  Processing: {txn_id}  (${transaction['amount']:,.2f})\n{'─'*65}")

        self._log("\n[STEP 1] Running ML anomaly scoring…")
        score, features = self._ml_score(transaction)
        self._log(f"[STEP 1] Score: {score:.4f}  [{self._risk_level(score)}]")

        explanation = self._llm_explain(transaction, score, features)
        self._log(f"[STEP 2] Done ({len(explanation)} chars)")

        decision = self._agent_decide(transaction, score, explanation)
        self._log(f"[STEP 3] Decision: {decision['action']}")

        return self._generate_report(transaction, score, features, explanation, decision)

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= RISK_THRESHOLDS["BLOCK"]:   return "CRITICAL"
        if score >= RISK_THRESHOLDS["FLAG"]:    return "HIGH"
        if score >= RISK_THRESHOLDS["MONITOR"]: return "MEDIUM"
        return "LOW"
