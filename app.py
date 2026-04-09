"""
AI Fraud Detection Agent — Streamlit UI
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0e1117; }

/* Decision badge */
.badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 8px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-align: center;
    width: 100%;
}
.badge-block   { background: #ff4b4b22; color: #ff4b4b; border: 2px solid #ff4b4b; }
.badge-flag    { background: #ffa50022; color: #ffa500; border: 2px solid #ffa500; }
.badge-monitor { background: #ffd70022; color: #ffd700; border: 2px solid #ffd700; }
.badge-approve { background: #00cc6622; color: #00cc66; border: 2px solid #00cc66; }

/* Step card */
.step-card {
    background: #1a1d27;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #2d3148;
}
.step-done   { border-color: #00cc66; }
.step-active { border-color: #4a9eff; }
.step-wait   { opacity: 0.4; }

/* Metric card */
.metric-box {
    background: #1a1d27;
    border-radius: 10px;
    padding: 18px;
    border: 1px solid #2d3148;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── cached model init ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    from agent import FraudDetectionAgent
    return FraudDetectionAgent(verbose=False)


# ── helpers ───────────────────────────────────────────────────────────────────
ACTION_STYLE = {
    "BLOCK":          ("badge-block",   "🔴 BLOCK"),
    "FLAG_FOR_REVIEW":("badge-flag",    "🟡 FLAG FOR REVIEW"),
    "MONITOR":        ("badge-monitor", "🟠 MONITOR"),
    "APPROVE":        ("badge-approve", "🟢 APPROVE"),
}

RISK_COLOR = {
    "CRITICAL": "#ff4b4b",
    "HIGH":     "#ffa500",
    "MEDIUM":   "#ffd700",
    "LOW":      "#00cc66",
}

def risk_level(score: float) -> str:
    if score >= 0.75: return "CRITICAL"
    if score >= 0.50: return "HIGH"
    if score >= 0.25: return "MEDIUM"
    return "LOW"

def score_gauge(score: float) -> go.Figure:
    color = RISK_COLOR[risk_level(score)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar":  {"color": color},
            "bgcolor": "#1a1d27",
            "bordercolor": "#2d3148",
            "steps": [
                {"range": [0,  25], "color": "rgba(0,204,102,0.08)"},
                {"range": [25, 50], "color": "rgba(255,215,0,0.08)"},
                {"range": [50, 75], "color": "rgba(255,165,0,0.08)"},
                {"range": [75,100], "color": "rgba(255,75,75,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
        title={"text": "Anomaly Score", "font": {"color": "#aaa", "size": 14}},
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0e1117",
        font_color="#ccc",
    )
    return fig

def feature_chart(features: dict) -> go.Figure:
    top = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = [k.replace("_", " ") for k, _ in top]
    values = [v for _, v in top]
    colors = ["#ff4b4b" if v > 1 else "#4a9eff" if v > 0 else "#555" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#ccc", "size": 11},
    ))
    fig.update_layout(
        height=340,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d27",
        xaxis=dict(gridcolor="#2d3148", color="#555", zeroline=True, zerolinecolor="#555"),
        yaxis=dict(gridcolor="#2d3148", color="#ccc", autorange="reversed"),
        font_color="#ccc",
    )
    return fig


# ── session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── sidebar — input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.markdown("---")
    st.markdown("### Transaction Input")

    txn_id   = st.text_input("Transaction ID", value="TXN_001")
    amount   = st.number_input("Amount ($)", min_value=1.0, value=9450.0, step=100.0, format="%.2f")
    sender   = st.text_input("Sender Account", value="ACCT_SENDER_001")
    receiver = st.text_input("Receiver Account", value="ACCT_RECV_001")

    country = st.selectbox("Country", [
        "US", "GB", "DE", "CA", "AU", "FR", "JP",
        "PA", "KY", "VG", "CY", "LB",   # high-risk
    ])
    category = st.selectbox("Category", [
        "transfer", "deposit", "retail", "food",
        "travel", "utilities", "healthcare", "payroll",
    ])
    date = st.date_input("Date", value=datetime(2025, 3, 20))
    time = st.time_input("Time", value=datetime(2025, 3, 20, 2, 45).time())

    st.markdown("---")
    run_btn = st.button("🔍 Analyze Transaction", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("#### Quick Load")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Clean", use_container_width=True):
        st.session_state["preset"] = "clean"
        st.rerun()
    if col2.button("⚠️ Suspect", use_container_width=True):
        st.session_state["preset"] = "suspect"
        st.rerun()
    if col3.button("🚨 Fraud", use_container_width=True):
        st.session_state["preset"] = "fraud"
        st.rerun()

# Handle presets
if "preset" in st.session_state:
    preset = st.session_state.pop("preset")
    presets = {
        "clean":   dict(txn_id="TXN_CLEAN", amount=87.45, sender="ACCT_ALICE", receiver="ACCT_SHOP", country="US", category="retail"),
        "suspect": dict(txn_id="TXN_STRUCT", amount=9450.0, sender="STRUCT_777", receiver="BANK_03", country="US", category="deposit"),
        "fraud":   dict(txn_id="TXN_FRAUD", amount=48000.0, sender="OFFSHORE_X", receiver="SHELL_PA", country="PA", category="transfer"),
    }
    for k, v in presets[preset].items():
        st.session_state[k] = v
    st.rerun()


# ── main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ AI Fraud Detection Agent")
st.markdown("*IsolationForest ML · Groq LLM · Autonomous Action Engine*")
st.markdown("---")

if not run_btn:
    # ── welcome / idle state ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, num, title, desc in [
        (c1, "①", "ML Scoring",    "IsolationForest anomaly detection"),
        (c2, "②", "LLM Analysis",  "Claude explains the anomaly"),
        (c3, "③", "Agent Decision","Autonomous BLOCK / FLAG / MONITOR"),
        (c4, "④", "Report",        "Structured JSON saved to disk"),
    ]:
        col.markdown(f"""
        <div class="step-card step-wait">
            <div style="font-size:2rem">{num}</div>
            <div style="font-weight:700;margin:6px 0">{title}</div>
            <div style="font-size:0.8rem;color:#888">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 Fill in the transaction details in the sidebar and click **Analyze Transaction**.")

    if st.session_state.history:
        st.markdown("### 📋 Analysis History")
        _df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            _df.style.applymap(
                lambda v: f"color: {RISK_COLOR.get(v, '#ccc')}",
                subset=["Risk Level"]
            ),
            use_container_width=True,
            hide_index=True,
        )

else:
    # ── run pipeline ──────────────────────────────────────────────────────────
    txn = {
        "txn_id":    txn_id,
        "sender":    sender,
        "receiver":  receiver,
        "amount":    float(amount),
        "country":   country,
        "category":  category,
        "timestamp": datetime.combine(date, time),
    }

    # Step progress header
    step_cols = st.columns(4)
    step_labels = ["① ML Scoring", "② LLM Analysis", "③ Agent Decision", "④ Report"]
    step_placeholders = [c.empty() for c in step_cols]

    def render_steps(active: int):
        for i, (ph, label) in enumerate(zip(step_placeholders, step_labels)):
            if i < active:
                cls = "step-card step-done"
                icon = "✅"
            elif i == active:
                cls = "step-card step-active"
                icon = "⏳"
            else:
                cls = "step-card step-wait"
                icon = "○"
            ph.markdown(f"""
            <div class="{cls}">
                <div style="font-size:1.4rem">{icon}</div>
                <div style="font-weight:600;font-size:0.9rem;margin-top:4px">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    render_steps(0)

    try:
        agent = load_agent()
    except Exception as e:
        st.error(f"Failed to load agent: {e}")
        st.stop()

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    with st.spinner("Running ML anomaly scoring…"):
        score, features = agent._ml_score(txn)

    render_steps(1)

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.plotly_chart(score_gauge(score), use_container_width=True)
        rl = risk_level(score)
        st.markdown(f"""
        <div class="metric-box">
            <div style="color:#888;font-size:0.85rem">RISK LEVEL</div>
            <div style="color:{RISK_COLOR[rl]};font-size:1.8rem;font-weight:700">{rl}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("**Top Feature Contributions**")
        st.plotly_chart(feature_chart(features), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Agent's Anomaly Explanation")
    explanation_box = st.empty()
    explanation_text = ""

    with st.spinner(""):
        for chunk in agent.stream_explanation(txn, score, features):
            explanation_text += chunk
            explanation_box.markdown(
                f'<div style="background:#1a1d27;border-radius:8px;padding:16px;'
                f'border:1px solid #2d3148;line-height:1.7">{explanation_text}▌</div>',
                unsafe_allow_html=True,
            )
    explanation_box.markdown(
        f'<div style="background:#1a1d27;border-radius:8px;padding:16px;'
        f'border:1px solid #2d3148;line-height:1.7">{explanation_text}</div>',
        unsafe_allow_html=True,
    )

    render_steps(2)

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    with st.spinner("Agent deciding action…"):
        decision = agent._agent_decide(txn, score, explanation_text)

    render_steps(3)

    action = decision["action"]
    badge_cls, badge_label = ACTION_STYLE.get(action, ("badge-flag", action))

    st.markdown("---")
    st.markdown("### ⚡ Agent Decision")
    st.markdown(f'<div class="badge {badge_cls}">{badge_label}</div>', unsafe_allow_html=True)
    st.markdown("")

    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        st.markdown(f"**Reason:** {decision.get('reason', '')}")
        if "analyst_notes" in decision:
            st.markdown(f"**Analyst Notes:** {decision['analyst_notes']}")
        if "key_signals" in decision:
            st.markdown("**Key Signals:**")
            for sig in decision["key_signals"]:
                st.markdown(f"- {sig}")
    with dcol2:
        if "duration_days" in decision:
            st.metric("Monitor Duration", f"{decision['duration_days']} days")
        st.metric("Risk Score", f"{score:.4f}")

    # ── STEP 4 ────────────────────────────────────────────────────────────────
    report = agent._generate_report(txn, score, features, explanation_text, decision)
    render_steps(4)

    st.markdown("---")
    with st.expander("📄 Full JSON Report"):
        import json
        st.code(json.dumps(report, indent=2), language="json")

    # Save to history
    st.session_state.history.append({
        "TXN ID":    txn_id,
        "Amount":    f"${amount:,.2f}",
        "Country":   country,
        "ML Score":  round(score, 4),
        "Risk Level": risk_level(score),
        "Decision":  action,
        "Time":      datetime.now().strftime("%H:%M:%S"),
    })

    st.success("✅ Analysis complete — report saved to reports/")

    # History at bottom
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 📋 Session History")
        _df = pd.DataFrame(st.session_state.history)
        st.dataframe(_df, use_container_width=True, hide_index=True)
