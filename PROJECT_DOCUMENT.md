# AI Fraud Detection Agent System
### Project Documentation & Interview Guide

---

## 1. Project Overview

An end-to-end AI agent that analyzes financial transactions for fraud using a hybrid approach: a classical Machine Learning model for anomaly detection, a Large Language Model for reasoning and decision-making, and a Streamlit dashboard for visualization.

**Tech Stack**

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn IsolationForest |
| Feature Engineering | Pandas, NetworkX (graph features) |
| LLM / Agent | Groq API — LLaMA-3.3-70B |
| UI | Streamlit + Plotly |
| Data | Synthetic AML dataset (smurfing, layering, structuring, round-tripping) |

---

## 2. System Architecture

```
INPUT: Transaction Dict
  { txn_id, sender, receiver, amount, country, category, timestamp }
          │
          ▼
┌─────────────────────────────────┐
│  FraudDetectionAgent.run()      │  ← Entry point (agent.py)
└─────────────────────────────────┘
          │
          ├──► STEP 1: _ml_score()
          │         │
          │         ├─ Appends transaction to context dataset
          │         ├─ Calls build_features() → 21 numeric features
          │         ├─ Scales with pre-fitted StandardScaler
          │         └─ Returns anomaly_score ∈ [0, 1]
          │
          ├──► STEP 2: _llm_explain()  /  stream_explanation()
          │         │
          │         ├─ Builds prompt with score + top 8 features
          │         ├─ Calls Groq API (LLaMA-3.3-70B)
          │         └─ Returns natural language explanation
          │
          ├──► STEP 3: _agent_decide()
          │         │
          │         ├─ Builds decision prompt with score + explanation
          │         ├─ Calls Groq with tool_choice="required"
          │         ├─ LLM selects one of 4 tools:
          │         │     block_transaction   → BLOCK        (score ≥ 0.75)
          │         │     flag_for_review     → FLAG         (score 0.50–0.74)
          │         │     monitor_account     → MONITOR      (score 0.25–0.49)
          │         │     approve_transaction → APPROVE      (score < 0.25)
          │         └─ Returns structured decision dict
          │
          └──► STEP 4: _generate_report()
                    │
                    ├─ Assembles full JSON report
                    ├─ Prints formatted console summary
                    └─ Saves to reports/{txn_id}_{timestamp}.json

OUTPUT: Report Dict + JSON file on disk
```

---

## 3. Function-by-Function Breakdown

### `__init__(verbose)`
- Reads `GROQ_API_KEY` from `.env`
- Initializes Groq client
- Calls `_train_model()`

### `_train_model()`
- Calls `generate_dataset()` → synthetic transactions with 4 laundering patterns
- Stratified sample: 1,200 legit + all fraud rows
- Calls `build_features()` → fits StandardScaler, stores `self._scaler`
- Trains `AMLIsolationForest(contamination=0.08)`
- Calls `tune_threshold(beta=2.0)` → optimizes F2-score (recall-weighted)
- Stores trained model as `self._model`

### `_ml_score(txn)`
- Converts transaction dict → DataFrame row
- Appends to `self._context_df` (so sender history / graph features are accurate)
- Calls `build_features()` on the augmented dataset
- Extracts last row (the new transaction)
- Returns `(score: float, features: dict)`

### `_explain_prompt(txn, score, features)`
- Builds structured prompt with transaction details + top 8 features sorted by magnitude
- Shared by both `_llm_explain()` and `stream_explanation()`

### `_llm_explain(txn, score, features)`
- Non-streaming — used by `demo.py`
- Single `chat.completions.create()` call to Groq
- Returns explanation string

### `stream_explanation(txn, score, features)`
- Streaming generator — used by `app.py` (Streamlit)
- `stream=True` on Groq API
- Yields text chunks for real-time display

### `_agent_decide(txn, score, explanation)`
- Calls Groq with `tools=DECISION_TOOLS` and `tool_choice="required"`
- LLM picks exactly one of 4 tools based on risk context
- Parses `tc.function.arguments` → JSON dict
- Returns `{ "action": "BLOCK/FLAG/MONITOR/APPROVE", ...args }`

### `_generate_report(txn, score, features, explanation, decision)`
- Assembles structured report dict
- Prints formatted console output with emoji decision icons
- Saves JSON to `reports/` folder with timestamp
- Returns report dict

### `run(transaction)`
- Public entry point — orchestrates all 4 steps in sequence
- Returns final report dict

### `_risk_level(score)` *(static)*
- Maps float score to label: CRITICAL / HIGH / MEDIUM / LOW

---

## 4. Feature Engineering (21 Features)

Imported from `08 AML/aml_detection/src/feature_engineering.py`

| Category | Features |
|---|---|
| **Behavioral** | amount, sender_txn_count, sender_mean_amt, sender_std_amt, sender_max_amt, sender_total_vol |
| **Receiver** | recv_txn_count, recv_mean_amt, recv_total_vol |
| **Ratio** | amt_vs_sender_mean |
| **Risk Flags** | high_risk_country, near_ctr_threshold |
| **Temporal** | hour, is_night |
| **Graph** | in_degree, out_degree, pagerank, scc_size, local_clustering |
| **Velocity** | velocity_1h, velocity_24h |

---

## 5. Money Laundering Patterns in Training Data

| Pattern | Description | Key Signal |
|---|---|---|
| **Smurfing** | Many small transactions split from one large amount | High volume, low individual amounts |
| **Layering** | Rapid multi-hop movement through accounts | High velocity, many hops |
| **Structuring** | Repeated deposits just under $10,000 CTR threshold | near_ctr_threshold = 1 |
| **Round-tripping** | Funds leave and return to same account | High SCC size (strongly connected component) |

---

## 6. UI — app.py (Streamlit)

```
Sidebar                          Main Panel
───────────────────              ──────────────────────────────────────
• Transaction input form         • Step progress cards (① ② ③ ④)
• Quick load presets             • Plotly anomaly score gauge
  ✅ Clean                       • Feature importance bar chart
  ⚠️  Suspect                    • LLM explanation (streaming live)
  🚨 Fraud                       • Color-coded decision badge
                                 • Key signals + analyst notes
                                 • Full JSON report (expandable)
                                 • Session history table
```

---

## 7. Data Flow Diagram

```
console.groq.com ──► GROQ_API_KEY ──► .env ──► agent.py
                                                    │
08 AML/src/ ────────────────────────────────────────┤
  data_generator.py    → synthetic transactions      │
  feature_engineering.py → 21 features               │
  isolation_forest_model.py → AMLIsolationForest     │
                                                    │
Transaction Input ──► _ml_score() ──► score [0,1]  │
                          │                         │
                          ▼                         │
                   _llm_explain() ──► Groq API      │
                          │                         │
                          ▼                         │
                   _agent_decide() ──► Groq API (tools)
                          │
                          ▼
                   _generate_report() ──► reports/*.json
```

---

---

# Interview Questions & Answers

---

## Section A — Machine Learning

**Q1. Why did you use IsolationForest for fraud detection?**

IsolationForest is an unsupervised anomaly detection algorithm — it doesn't need labeled fraud examples to train, which is realistic because in real AML systems, labeled data is rare and expensive. It works by randomly partitioning data; anomalous points (outliers) are isolated faster with fewer splits, giving them a higher anomaly score. It also scales well to high-dimensional data and handles class imbalance naturally since it doesn't rely on class boundaries.

---

**Q2. What is the contamination parameter and how did you set it?**

Contamination is the expected proportion of anomalies in the training data. We set it to `0.08` (8%) which roughly matches the fraud rate in our synthetic dataset (~8–10% laundering transactions). Setting it too low makes the model miss fraud; too high causes false positives. In production, you'd estimate this from historical confirmed fraud rates.

---

**Q3. What does tune_threshold(beta=2.0) do and why beta=2?**

It sweeps precision-recall thresholds on validation data and picks the one that maximizes F-beta score. Beta=2 weights recall twice as much as precision, meaning we'd rather have false positives (flag a legit transaction) than false negatives (miss real fraud). In AML, missing fraud is far more costly than over-flagging.

---

**Q4. Why do you append the new transaction to the context dataset before scoring?**

Because many features are relative — `amt_vs_sender_mean`, `sender_txn_count`, graph centrality, and velocity features all require historical context about the sender's behavior. Without appending, the new transaction would be scored as if the sender has no history, making these features meaningless. Appending gives us accurate contextual features.

---

**Q5. What are the 3 categories of features and why does each matter?**

- **Behavioral:** Captures sender/receiver history — how much does this transaction deviate from the sender's normal patterns?
- **Graph:** Uses NetworkX to compute PageRank, betweenness, SCC size — laundering rings form detectable graph structures (e.g., round-tripping creates strongly connected components).
- **Velocity/Temporal:** High transaction frequency in a short window is a classic smurfing/layering signal; night-time activity correlates with suspicious behavior.

---

**Q6. How would you handle model drift in production?**

Monitor the anomaly score distribution over time. If legitimate transactions start scoring higher (distribution shift), retrain on a rolling window of recent data. You'd also track precision/recall on confirmed fraud cases, set up alerts if recall drops below a threshold, and potentially use online learning or periodic retraining schedules.

---

## Section B — LLM / Agent Design

**Q7. Why use an LLM at all if the ML model already gives a score?**

The ML score tells you *that* something is anomalous, not *why*. The LLM converts raw feature values into human-readable reasoning that analysts can act on. It also adds domain knowledge — identifying *which* money laundering pattern applies requires reasoning that a pure numerical model can't provide. The combination of ML speed + LLM reasoning is more useful than either alone.

---

**Q8. What is tool use / function calling and why did you use it for the decision step?**

Tool use forces the LLM to respond in a structured, machine-parseable format by selecting from a predefined set of functions (tools). Instead of generating free-form text like "I think you should block this," the model must call `block_transaction(reason=..., risk_score=..., key_signals=[...])`. This gives us:
- Guaranteed structured output (no text parsing needed)
- Type safety (risk_score is always a number)
- Constrained action space (model can only pick from 4 valid actions)

---

**Q9. What is `tool_choice="required"` and why use it?**

It forces the LLM to call a tool rather than optionally responding with text. Without it, the model might respond with "I recommend blocking this transaction" as plain text instead of calling `block_transaction()`. `required` guarantees we always get a structured decision, making the pipeline deterministic and parseable.

---

**Q10. Why did you separate the explanation step from the decision step?**

Single responsibility — the explanation step focuses on *understanding* (what patterns are present), while the decision step focuses on *action* (what to do about it). Passing the explanation as context to the decision step gives the agent richer reasoning input than just the raw score. It also makes debugging easier and allows each step to be tested independently.

---

**Q11. What is streaming and why use it in the UI?**

Streaming sends the LLM response token by token as it's generated, rather than waiting for the complete response. This dramatically improves perceived performance — the user sees text appearing immediately rather than waiting 3–5 seconds for a blank screen. In Streamlit, `stream=True` on the Groq API + `yield chunk` in `stream_explanation()` feeds into real-time UI updates.

---

**Q12. How would you improve the agent if you had production resources?**

1. Add a **memory/database** of past decisions so the agent can detect repeat offenders
2. Add a **human-in-the-loop** approval step for BLOCK decisions
3. Use **RAG** (Retrieval-Augmented Generation) over a knowledge base of AML typologies
4. Add **multi-agent** structure: separate agents for pattern detection, regulatory checking, and final decision
5. Replace LLaMA with a fine-tuned model on real AML investigation reports

---

## Section C — System Design

**Q13. How would you scale this to handle 1 million transactions per day?**

- Run ML scoring in **batch** using vectorized numpy operations (no per-transaction loop)
- Use a **message queue** (Kafka/RabbitMQ) to buffer incoming transactions
- Call the LLM only for transactions above a score threshold (e.g., > 0.3) to reduce API calls by ~80%
- Cache explanations for similar feature vectors using embedding similarity
- Deploy the ML model as a **microservice** separate from the LLM agent

---

**Q14. What are the risks of automating BLOCK decisions?**

- **False positives** could block legitimate customers, causing financial and reputational damage
- **Adversarial attacks** — fraudsters could learn to manipulate features to stay below threshold
- **Regulatory risk** — automated blocking must comply with AML regulations and provide audit trails
- Mitigation: use BLOCK only for very high confidence (≥ 0.75), always log decisions with reasons, maintain human override capability

---

**Q15. How does the Streamlit caching (`@st.cache_resource`) work here?**

`@st.cache_resource` caches the `FraudDetectionAgent` object across Streamlit reruns. Without it, every button click would retrain the IsolationForest from scratch (~10–15 seconds). With it, the model is trained once on first load and reused for all subsequent transactions in the session. `cache_resource` is used (vs `cache_data`) because the agent holds a network connection (Groq client) that shouldn't be serialized.

---

**Q16. What would you add to the report for a real compliance team?**

- **SAR readiness fields**: case ID, investigator assignment, regulatory jurisdiction
- **Audit trail**: who reviewed, what action was taken, timestamps
- **Evidence links**: transaction graph visualization, related account network
- **Regulatory mappings**: which FATF typology, which AML rule was triggered
- **Confidence intervals**: not just a score but a probability range

---

## Section D — Quick-Fire Definitions

| Term | One-line Answer |
|---|---|
| **Anomaly Score** | How different a transaction is from the training distribution (0=normal, 1=anomalous) |
| **Contamination** | Expected fraud rate in training data; affects where IsolationForest sets its decision boundary |
| **F-beta Score** | Weighted harmonic mean of precision and recall; beta>1 favors recall |
| **PageRank** | Graph metric measuring how many important nodes point to an account; high PageRank = central hub in transaction network |
| **SCC** | Strongly Connected Component — a group of accounts where every account can reach every other; indicates round-tripping |
| **CTR Threshold** | $10,000 Cash Transaction Report limit; structuring means breaking amounts just below this |
| **Tool Use** | Structured LLM output format where the model calls predefined functions instead of generating free text |
| **Groq** | Inference provider using custom LPU chips; offers free-tier LLaMA API access with high speed |
| **Streamlit** | Python library for building data dashboards with minimal frontend code |
| **Velocity Feature** | Count of transactions by the same sender within a time window (1h, 24h) |
