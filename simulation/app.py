"""
TA-BC-BFL Framework Simulator
Trust-Aware Blockchain-Enabled Bio-Aware Federated Learning
Interactive Streamlit Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os

# Ensure the simulation package is importable when running via `streamlit run simulation/app.py`
_sim_dir = os.path.dirname(os.path.abspath(__file__))
if _sim_dir not in os.sys.path:
    os.sys.path.insert(0, _sim_dir)

from framework.federated_learning import FederatedLearningSimulator, SimulationConfig
from framework.bio_aware import OMICS_TYPES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TA-BC-BFL Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #6366f1; }
.metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.tag-green { background: #064e3b; color: #6ee7b7; }
.tag-red { background: #450a0a; color: #fca5a5; }
.tag-blue { background: #1e3a5f; color: #93c5fd; }
.tag-yellow { background: #451a03; color: #fde68a; }
.block-hash { font-family: monospace; font-size: 0.7rem; color: #64748b; word-break: break-all; }
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
    border-bottom: 1px solid #334155; padding-bottom: 8px; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "TA-BC-BFL":     "#6366f1",
    "FEDAVG":        "#94a3b8",
    "BLOCKCHAIN-FL": "#f59e0b",
    "TRUST-FL":      "#10b981",
}
OMICS_COLORS = {
    "genomic":       "#6366f1",
    "proteomic":     "#f59e0b",
    "clinical":      "#10b981",
    "epigenomic":    "#ef4444",
    "transcriptomic":"#8b5cf6",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR – Simulation Parameters
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 TA-BC-BFL Simulator")
    st.markdown("*Trust-Aware Blockchain-Enabled Bio-Aware Federated Learning*")
    st.divider()

    st.markdown("### 🏥 Federation Setup")
    num_clients = st.slider("Number of Clients (Hospitals)", 4, 20, 10)
    num_rounds  = st.slider("Communication Rounds", 5, 50, 20)
    local_epochs = st.slider("Local Epochs per Round", 1, 10, 5)
    fraction_clients = st.slider("Client Participation Rate", 0.3, 1.0, 1.0, 0.1)
    num_samples = st.slider("Samples per Client", 200, 2000, 500, 100)

    st.divider()
    st.markdown("### 🧬 Bio-Aware Aggregation")
    bio_weight     = st.slider("Bio-Similarity Weight", 0.0, 1.0, 0.4, 0.05)
    sample_weight  = st.slider("Sample-Count Weight",   0.0, 1.0, 0.3, 0.05)
    quality_weight = st.slider("Quality Score Weight",  0.0, 1.0, 0.3, 0.05)
    bio_temp       = st.slider("Softmax Temperature",   0.5, 3.0, 1.0, 0.1)

    data_dist = st.selectbox(
        "Data Distribution",
        ["non_iid_mild", "non_iid_severe", "iid"],
        format_func=lambda x: {"non_iid_mild": "Non-IID (Mild)", "non_iid_severe": "Non-IID (Severe)", "iid": "IID"}[x],
    )

    st.divider()
    st.markdown("### ⛓️ Blockchain Trust")
    initial_trust  = st.slider("Initial Trust Score", 0.3, 1.0, 0.7, 0.05)
    trust_lr       = st.slider("Trust Learning Rate", 0.05, 0.5, 0.15, 0.05)
    anomaly_thr    = st.slider("Anomaly Threshold (σ)", 1.0, 5.0, 2.5, 0.25)
    mining_diff    = st.selectbox("Mining Difficulty", [1, 2, 3], index=1)

    st.divider()
    st.markdown("### 🔒 Privacy")
    use_dp      = st.toggle("Differential Privacy", value=True)
    dp_epsilon  = st.slider("DP Epsilon (ε)", 0.1, 10.0, 1.0, 0.1, disabled=not use_dp)
    use_he      = st.toggle("Homomorphic Encryption", value=True)
    he_key_size = st.selectbox("HE Key Size (bits)", [1024, 2048, 4096], index=1, disabled=not use_he)

    st.divider()
    st.markdown("### ⚔️ Adversarial Setting")
    adv_fraction = st.slider("Adversarial Client Fraction", 0.0, 0.5, 0.0, 0.05)
    attack_type  = st.selectbox(
        "Attack Type",
        ["model_poisoning", "label_flipping", "sybil"],
        format_func=lambda x: {"model_poisoning": "Model Poisoning", "label_flipping": "Label Flipping", "sybil": "Sybil Attack"}[x],
        disabled=adv_fraction == 0,
    )
    attack_scale = st.slider("Attack Intensity", 1.0, 10.0, 3.0, 0.5, disabled=adv_fraction == 0)

    st.divider()
    run_baselines = st.toggle("Compare with Baselines", value=True)

    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧬 TA-BC-BFL Framework Simulator")
st.markdown(
    "**Trust-Aware Blockchain-Enabled Bio-Aware Federated Learning** for "
    "secure, scalable, and privacy-preserving healthcare data integration."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_sim, tab_results, tab_trust, tab_privacy, tab_attacks, tab_chain = st.tabs([
    "📖 Overview",
    "▶️ Simulation",
    "📊 Results",
    "🛡️ Trust Scores",
    "🔒 Privacy",
    "⚔️ Attacks",
    "⛓️ Blockchain",
])

# ══════════════════════════════════════════════════════
#  TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## Framework Architecture")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Core Paradigm</div><div class="metric-value" style="font-size:1.2rem;">Federated Learning</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">Trust Layer</div><div class="metric-value" style="font-size:1.2rem;">Blockchain</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">Aggregation</div><div class="metric-value" style="font-size:1.2rem;">Bio-Aware</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Privacy</div><div class="metric-value" style="font-size:1.2rem;">DP + HE</div></div>', unsafe_allow_html=True)

    st.divider()

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### Key Innovations")
        innovations = [
            ("🧬 Bio-Aware Aggregation",
             "Multi-omics pathway similarity replaces simple sample-weighted averaging. "
             "Clients sharing biological context contribute more to the global model."),
            ("⛓️ Adaptive Trust Scoring",
             "Blockchain records each client's gradient quality per round. "
             "Trust scores dynamically adjust — malicious updates are down-weighted automatically."),
            ("🔒 Privacy Stack",
             "Differential Privacy (Gaussian mechanism) bounds individual data leakage. "
             "Homomorphic Encryption allows aggregation without decrypting raw gradients."),
            ("🛡️ Adversarial Robustness",
             "Combines trust scoring with anomaly detection to resist model poisoning, "
             "Sybil attacks, gradient inversion, and backdoor injections."),
            ("📡 Scalable Architecture",
             "Fraction-based client selection and lightweight blockchain mining "
             "enable deployment across hundreds of hospital nodes."),
        ]
        for title, desc in innovations:
            with st.expander(title, expanded=True):
                st.write(desc)

    with col_r:
        st.markdown("### Omics Types Supported")
        omics_df = pd.DataFrame([
            {"Type": t, "Dimension": d, "Examples": e}
            for t, d, e in [
                ("Genomic",       "~20K genes",    "SNPs, CNVs, mutations"),
                ("Proteomic",     "~10K proteins", "Protein expression"),
                ("Clinical",      "~50–500",       "Lab results, vitals"),
                ("Epigenomic",    "~5K marks",     "Methylation, chromatin"),
                ("Transcriptomic","~15K transcripts","mRNA expression, splicing"),
            ]
        ])
        st.dataframe(omics_df, hide_index=True, use_container_width=True)

        st.markdown("### Research Questions (from paper)")
        for rq in [
            "**RQ1** How does bio-similarity improve model generalization?",
            "**RQ2** Can blockchain trust scoring mitigate adversarial attacks?",
            "**RQ3** What is the DP privacy–accuracy trade-off?",
        ]:
            st.markdown(f"- {rq}")

    st.divider()
    st.markdown("### Aggregation Formula")
    st.latex(r"""
    w_i^{(\text{final})} = \text{softmax}\!\left(
        \alpha \cdot \bar{s}_i^{\text{bio}} +
        \beta  \cdot \frac{n_i}{\sum_j n_j} +
        \gamma \cdot q_i
    \right) \cdot \tau_i
    """)
    st.caption(
        r"where $\bar{s}_i^{\text{bio}}$ = average bio similarity, "
        r"$n_i$ = sample count, $q_i$ = quality score, $\tau_i$ = blockchain trust score, "
        r"$\alpha+\beta+\gamma=1$."
    )

# ══════════════════════════════════════════════════════
#  TAB 2 – SIMULATION (Run)
# ══════════════════════════════════════════════════════
with tab_sim:
    if not run_btn and "sim_results" not in st.session_state:
        st.info("👈 Configure simulation parameters in the sidebar, then click **Run Simulation**.")

        st.markdown("### What gets simulated")
        steps = [
            ("1. Client Initialization", f"Create {num_clients} hospital clients with heterogeneous multi-omics data ({data_dist} distribution)."),
            ("2. Local Training", f"Each client trains a logistic regression model for {local_epochs} epoch(s) per round without sharing raw data."),
            ("3. Privacy Application", f"{'Gaussian DP noise (ε='+str(dp_epsilon)+') + ' if use_dp else ''}{'Homomorphic encryption ' if use_he else ''}applied to gradients before transmission."),
            ("4. Blockchain Trust Update", f"Each round creates a new block. Gradient anomaly detection updates trust scores (initial={initial_trust})."),
            ("5. Bio-Aware Aggregation", f"Multi-omics pathway similarity + trust scores compute final aggregation weights (α={bio_weight}, β={sample_weight}, γ={quality_weight})."),
            ("6. Evaluation", f"Global model evaluated on all clients after each of {num_rounds} rounds."),
        ]
        for title, desc in steps:
            st.markdown(f"**{title}**: {desc}")

    if run_btn:
        config = SimulationConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=0.01,
            fraction_clients=fraction_clients,
            global_model_dim=64,
            bio_weight=bio_weight,
            sample_weight=sample_weight,
            quality_weight=quality_weight,
            bio_temperature=bio_temp,
            initial_trust=initial_trust,
            trust_learning_rate=trust_lr,
            trust_decay=0.95,
            anomaly_threshold=anomaly_thr,
            mining_difficulty=mining_diff,
            use_dp=use_dp,
            dp_epsilon=dp_epsilon if use_dp else 1.0,
            dp_delta=1e-5,
            use_he=use_he,
            he_key_size=he_key_size if use_he else 2048,
            adversarial_fraction=adv_fraction,
            attack_type=attack_type,
            attack_scale=attack_scale,
            num_samples_per_client=num_samples,
            data_distribution=data_dist,
            run_baselines=run_baselines,
            baseline_methods=["fedavg", "perfedavg", "blockchain_fl", "trust_fl"] if run_baselines else [],
        )

        progress_bar = st.progress(0.0, "Initializing simulation…")
        status_text  = st.empty()
        log_box      = st.empty()
        log_lines: list[str] = []

        def update_progress(pct: float, msg: str):
            progress_bar.progress(min(pct, 1.0), msg)
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            log_box.code("\n".join(log_lines[-15:]))

        simulator = FederatedLearningSimulator(config, progress_callback=update_progress)

        t0 = time.time()
        all_results = simulator.run_all()
        elapsed = time.time() - t0

        progress_bar.progress(1.0, "✅ Simulation complete!")
        status_text.success(f"Simulation finished in {elapsed:.1f}s · {num_rounds} rounds · {num_clients} clients")

        st.session_state["sim_results"]  = all_results
        st.session_state["sim_config"]   = config
        st.session_state["simulator"]    = simulator
        st.session_state["sim_elapsed"]  = elapsed
        st.rerun()

    elif "sim_results" in st.session_state:
        all_results = st.session_state["sim_results"]
        config      = st.session_state["sim_config"]
        elapsed     = st.session_state["sim_elapsed"]
        tabfl       = all_results["TA-BC-BFL"]

        st.success(f"✅ Last simulation: {config.num_rounds} rounds, {config.num_clients} clients, {elapsed:.1f}s")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("TA-BC-BFL Accuracy", f"{tabfl.final_accuracy:.1%}")
        with c2:
            st.metric("TA-BC-BFL Loss", f"{tabfl.final_loss:.4f}")
        with c3:
            chain_ok = "✅ Valid" if tabfl.blockchain_integrity else "❌ Compromised"
            st.metric("Chain Integrity", chain_ok)
        with c4:
            n_blocks = len(st.session_state["simulator"].trust_manager.chain)
            st.metric("Blocks Mined", str(n_blocks))

        st.info("Switch to the **Results**, **Trust Scores**, **Privacy**, or **Blockchain** tabs for detailed analysis.")


# ══════════════════════════════════════════════════════
#  TAB 3 – RESULTS
# ══════════════════════════════════════════════════════
with tab_results:
    if "sim_results" not in st.session_state:
        st.info("Run the simulation first (▶️ Simulation tab).")
    else:
        all_results = st.session_state["sim_results"]
        config      = st.session_state["sim_config"]

        # ── Accuracy curves ───────────────────────────────────────────────
        st.markdown("### 📈 Accuracy Over Communication Rounds")
        fig_acc = go.Figure()
        for method, result in all_results.items():
            rounds = [r.round_number for r in result.rounds]
            accs   = [r.global_accuracy for r in result.rounds]
            color  = COLORS.get(method, "#cbd5e1")
            fig_acc.add_trace(go.Scatter(
                x=rounds, y=accs, mode="lines+markers",
                name=method, line=dict(color=color, width=3 if method == "TA-BC-BFL" else 2),
                marker=dict(size=5),
            ))
        fig_acc.update_layout(
            xaxis_title="Communication Round",
            yaxis_title="Global Accuracy",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_dark", height=400,
            yaxis=dict(tickformat=".1%"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # ── Loss curves ───────────────────────────────────────────────────
        st.markdown("### 📉 Loss Over Communication Rounds")
        fig_loss = go.Figure()
        for method, result in all_results.items():
            rounds = [r.round_number for r in result.rounds]
            losses = [r.global_loss for r in result.rounds]
            color  = COLORS.get(method, "#cbd5e1")
            fig_loss.add_trace(go.Scatter(
                x=rounds, y=losses, mode="lines+markers",
                name=method, line=dict(color=color, width=3 if method == "TA-BC-BFL" else 2),
                marker=dict(size=5),
            ))
        fig_loss.update_layout(
            xaxis_title="Communication Round",
            yaxis_title="Cross-Entropy Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # ── Final comparison bar chart ────────────────────────────────────
        st.markdown("### 🏆 Final Performance Comparison")
        methods   = list(all_results.keys())
        final_acc = [r.final_accuracy for r in all_results.values()]
        final_los = [r.final_loss for r in all_results.values()]

        fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Final Accuracy", "Final Loss"))
        colors_list = [COLORS.get(m, "#cbd5e1") for m in methods]
        fig_bar.add_trace(go.Bar(x=methods, y=final_acc, marker_color=colors_list, name="Accuracy", text=[f"{a:.1%}" for a in final_acc], textposition="outside"), row=1, col=1)
        fig_bar.add_trace(go.Bar(x=methods, y=final_los, marker_color=colors_list, name="Loss", text=[f"{l:.4f}" for l in final_los], textposition="outside"), row=1, col=2)
        fig_bar.update_layout(template="plotly_dark", height=380, showlegend=False)
        fig_bar.update_yaxes(tickformat=".1%", row=1, col=1)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────
        st.markdown("### 📋 Summary Table")
        tabfl = all_results.get("TA-BC-BFL")
        rows = []
        for method, result in all_results.items():
            tabfl_acc = tabfl.final_accuracy if tabfl else result.final_accuracy
            delta = (result.final_accuracy - tabfl_acc) if method != "TA-BC-BFL" else 0.0
            rows.append({
                "Method": method,
                "Final Accuracy": f"{result.final_accuracy:.1%}",
                "Final Loss": f"{result.final_loss:.4f}",
                "vs TA-BC-BFL": f"{delta:+.1%}" if method != "TA-BC-BFL" else "—",
                "Chain Integrity": "✅" if result.blockchain_integrity else "—",
                "Rounds": config.num_rounds,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Client omics distribution ─────────────────────────────────────
        st.markdown("### 🧬 Client Omics Distribution")
        simulator = st.session_state["simulator"]
        omics_counts = {}
        for client in simulator.clients:
            ot = client.config.omics_type
            omics_counts[ot] = omics_counts.get(ot, 0) + 1

        fig_pie = px.pie(
            values=list(omics_counts.values()),
            names=list(omics_counts.keys()),
            color=list(omics_counts.keys()),
            color_discrete_map=OMICS_COLORS,
            template="plotly_dark",
            title="Distribution of Multi-Omics Data Types Across Clients",
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════
#  TAB 4 – TRUST SCORES
# ══════════════════════════════════════════════════════
with tab_trust:
    if "sim_results" not in st.session_state:
        st.info("Run the simulation first.")
    else:
        simulator = st.session_state["simulator"]
        config    = st.session_state["sim_config"]
        tm        = simulator.trust_manager
        trust_matrix = tm.get_trust_history_matrix()  # shape: (rounds+1, n_clients)

        st.markdown("### 🛡️ Trust Score Evolution per Client")
        fig_trust = go.Figure()
        n_clients = trust_matrix.shape[1]
        for c in range(n_clients):
            client      = simulator.clients[c]
            is_adversarial = client.is_adversarial
            omics       = client.config.omics_type
            color       = OMICS_COLORS.get(omics, "#94a3b8")
            dash        = "dash" if is_adversarial else "solid"
            label       = f"C{c} ({omics[:3]})" + (" ⚠️" if is_adversarial else "")
            fig_trust.add_trace(go.Scatter(
                x=list(range(trust_matrix.shape[0])),
                y=trust_matrix[:, c],
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash=dash),
                opacity=0.85,
            ))
        fig_trust.update_layout(
            xaxis_title="Round", yaxis_title="Trust Score",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_dark", height=430,
            legend=dict(orientation="v", x=1.01),
        )
        st.plotly_chart(fig_trust, use_container_width=True)

        # ── Current trust bar ────────────────────────────────────────────
        st.markdown("### 🏅 Final Trust Scores")
        current_trust = tm.trust_scores
        fig_bar = go.Figure()
        trust_colors = [
            "#ef4444" if simulator.clients[c].is_adversarial
            else OMICS_COLORS.get(simulator.clients[c].config.omics_type, "#6366f1")
            for c in range(n_clients)
        ]
        fig_bar.add_trace(go.Bar(
            x=[f"C{c}" for c in range(n_clients)],
            y=current_trust,
            marker_color=trust_colors,
            text=[f"{t:.2f}" for t in current_trust],
            textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_title="Client", yaxis_title="Trust Score",
            yaxis=dict(range=[0, 1.15]),
            template="plotly_dark", height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Anomaly log ──────────────────────────────────────────────────
        if tm.anomalies_detected:
            st.markdown("### ⚠️ Anomalies Detected")
            anomaly_df = pd.DataFrame(tm.anomalies_detected)
            st.dataframe(anomaly_df.style.highlight_max(color="#7f1d1d", axis=0, subset=["grad_norm"]), hide_index=True, use_container_width=True)
        else:
            st.success("✅ No anomalies detected during simulation.")

        # ── Client trust summary ─────────────────────────────────────────
        st.markdown("### 📋 Client Trust Summary")
        trust_summary = tm.get_client_trust_summary()
        rows = []
        for cid, info in trust_summary.items():
            client = simulator.clients[cid]
            rows.append({
                "Client": f"C{cid}",
                "Omics Type": client.config.omics_type,
                "Adversarial": "⚠️ Yes" if client.is_adversarial else "No",
                "Current Trust": f"{info['current']:.3f}",
                "Min": f"{info['min']:.3f}",
                "Max": f"{info['max']:.3f}",
                "Trend": f"{info['trend']:+.3f}",
                "Flags": info["flags"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════
#  TAB 5 – PRIVACY
# ══════════════════════════════════════════════════════
with tab_privacy:
    if "sim_results" not in st.session_state:
        st.info("Run the simulation first.")
    else:
        simulator = st.session_state["simulator"]
        config    = st.session_state["sim_config"]
        tabfl     = st.session_state["sim_results"]["TA-BC-BFL"]

        col_dp, col_he = st.columns(2)

        # ── DP panel ──────────────────────────────────────────────────────
        with col_dp:
            st.markdown("### 🎲 Differential Privacy")
            if config.use_dp and simulator.dp:
                pr = simulator.dp.get_privacy_report()
                st.metric("Privacy Budget (ε)", f"{pr['epsilon']:.2f}")
                st.metric("Spent Budget", f"{pr['spent_epsilon']:.3f}")
                st.metric("Budget Used", f"{pr['budget_used_fraction']:.1%}")
                st.metric("Noise Multiplier (σ)", f"{pr['noise_multiplier']:.4f}")
                st.metric("Max Gradient Norm", f"{pr['max_grad_norm']:.2f}")
                st.metric("Rounds Completed", str(pr['rounds_completed']))

                # Privacy–accuracy trade-off visual
                epsilons = np.linspace(0.1, 10.0, 50)
                noise_mults = np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilons
                est_acc_drop = np.exp(-epsilons * 0.3) * 0.15

                fig_dp = go.Figure()
                fig_dp.add_trace(go.Scatter(x=epsilons, y=noise_mults, name="Noise Multiplier", yaxis="y1", line=dict(color="#ef4444", width=2)))
                fig_dp.add_trace(go.Scatter(x=epsilons, y=est_acc_drop, name="Est. Acc. Drop", yaxis="y2", line=dict(color="#f59e0b", width=2, dash="dash")))
                fig_dp.add_vline(x=config.dp_epsilon, line_color="#6366f1", line_dash="dot", annotation_text=f"ε={config.dp_epsilon}")
                fig_dp.update_layout(
                    title="Privacy–Accuracy Trade-off",
                    xaxis_title="Privacy Budget (ε)",
                    yaxis=dict(title="Noise Multiplier", side="left"),
                    yaxis2=dict(title="Est. Accuracy Drop", overlaying="y", side="right"),
                    template="plotly_dark", height=360,
                )
                st.plotly_chart(fig_dp, use_container_width=True)
            else:
                st.warning("Differential Privacy was disabled in this simulation run.")

        # ── HE panel ──────────────────────────────────────────────────────
        with col_he:
            st.markdown("### 🔐 Homomorphic Encryption")
            if config.use_he and simulator.he:
                he = simulator.he
                st.metric("Key Size", f"{he.key_size_bits} bits")
                st.metric("Security Level", he.get_security_level())
                st.metric("Precision Bits", str(he.precision_bits))
                st.metric("Encryption Ops", str(he.encryption_ops))
                st.metric("Decryption Ops", str(he.decryption_ops))
                st.metric("Aggregation Ops", str(he.aggregation_ops))

                # Communication overhead visualization
                overhead = he.total_ciphertext_size_bytes / max(1, config.num_clients * config.num_rounds)
                plaintext_size = config.global_model_dim * 8  # float64

                fig_comm = go.Figure(go.Waterfall(
                    x=["Plaintext Size", "HE Overhead", "Total Ciphertext"],
                    measure=["absolute", "relative", "total"],
                    y=[plaintext_size, overhead, 0],
                    text=[f"{plaintext_size:,}B", f"+{overhead:,.0f}B", f"{plaintext_size + overhead:,.0f}B"],
                    textposition="outside",
                    decreasing={"marker": {"color": "#10b981"}},
                    increasing={"marker": {"color": "#ef4444"}},
                    totals={"marker": {"color": "#6366f1"}},
                ))
                fig_comm.update_layout(
                    title="Avg. Communication Cost per Round",
                    yaxis_title="Bytes",
                    template="plotly_dark", height=360,
                )
                st.plotly_chart(fig_comm, use_container_width=True)
            else:
                st.warning("Homomorphic Encryption was disabled in this simulation run.")

        # ── Noise vs SNR history ──────────────────────────────────────────
        if config.use_dp and simulator.dp and simulator.dp.noise_history:
            st.markdown("### 📡 Signal-to-Noise Ratio History")
            snr_vals = [n["snr_db"] for n in simulator.dp.noise_history]
            fig_snr = go.Figure()
            fig_snr.add_trace(go.Scatter(y=snr_vals, mode="lines+markers", name="SNR (dB)", line=dict(color="#6366f1", width=2)))
            fig_snr.update_layout(
                xaxis_title="Encryption Operation",
                yaxis_title="SNR (dB)",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig_snr, use_container_width=True)


# ══════════════════════════════════════════════════════
#  TAB 6 – ATTACKS
# ══════════════════════════════════════════════════════
with tab_attacks:
    if "sim_results" not in st.session_state:
        st.info("Run the simulation first.")
    else:
        simulator = st.session_state["simulator"]
        config    = st.session_state["sim_config"]

        st.markdown("### ⚔️ Adversarial Robustness Analysis")

        n_adv = int(config.num_clients * config.adversarial_fraction)
        if n_adv == 0:
            st.info("No adversarial clients were configured. Set 'Adversarial Client Fraction' > 0 to test robustness.")

        # ── Robustness comparison ─────────────────────────────────────────
        st.markdown("#### Robustness Under Adversarial Conditions")
        all_results = st.session_state["sim_results"]

        robustness_data = []
        for method, result in all_results.items():
            all_accs = [r.global_accuracy for r in result.rounds]
            std_acc  = float(np.std(all_accs[-5:])) if len(all_accs) >= 5 else float(np.std(all_accs))
            robustness_data.append({
                "Method": method,
                "Final Accuracy": result.final_accuracy,
                "Stability (low σ = better)": std_acc,
                "Adversarial Clients": n_adv,
                "Attack Type": config.attack_type if n_adv > 0 else "None",
            })

        fig_rob = px.scatter(
            pd.DataFrame(robustness_data),
            x="Stability (low σ = better)",
            y="Final Accuracy",
            text="Method",
            size=[20] * len(robustness_data),
            color="Method",
            color_discrete_map=COLORS,
            template="plotly_dark",
            title="Accuracy vs. Stability Trade-off (lower σ = more robust)",
        )
        fig_rob.update_traces(textposition="top center")
        fig_rob.update_layout(height=400, yaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_rob, use_container_width=True)

        # ── Gradient inversion attack demo ────────────────────────────────
        st.markdown("#### 🔬 Gradient Inversion Attack Simulation")
        from framework.attacks import AdversarialAttackSimulator
        from framework.clients import LocalModelUpdate

        if st.button("Run Gradient Inversion Demo"):
            atk_sim = AdversarialAttackSimulator()
            dummy_update = LocalModelUpdate(
                client_id=0,
                weights=np.random.randn(64) * 0.1,
                gradients=np.random.randn(10) * 0.5,
                loss=0.3,
                accuracy=0.75,
                num_samples=500,
                omics_type="genomic",
                bio_features=np.random.randn(16),
                round_number=1,
            )
            result_no_dp  = atk_sim.gradient_inversion_attempt(dummy_update, iterations=200)

            # With DP noise applied
            dummy_dp = LocalModelUpdate(
                client_id=0,
                weights=dummy_update.weights.copy(),
                gradients=dummy_update.gradients + np.random.randn(*dummy_update.gradients.shape) * 2.0,
                loss=dummy_update.loss,
                accuracy=dummy_update.accuracy,
                num_samples=dummy_update.num_samples,
                omics_type=dummy_update.omics_type,
                bio_features=dummy_update.bio_features.copy(),
                round_number=1,
                noise_applied=True,
            )
            result_with_dp = atk_sim.gradient_inversion_attempt(dummy_dp, iterations=200)

            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(y=result_no_dp["reconstruction_quality_history"], name="No DP (vulnerable)", line=dict(color="#ef4444", width=2)))
            fig_inv.add_trace(go.Scatter(y=result_with_dp["reconstruction_quality_history"], name="With DP (protected)", line=dict(color="#10b981", width=2)))
            fig_inv.update_layout(
                title="Gradient Inversion Attack: Reconstruction Quality Over Iterations",
                xaxis_title="Iteration",
                yaxis_title="Reconstruction Quality (higher = more data leaked)",
                template="plotly_dark", height=380,
            )
            st.plotly_chart(fig_inv, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                risk = result_no_dp["data_leakage_risk"]
                color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
                st.markdown(f'<span class="tag tag-red">Without DP – Data Leakage Risk: {risk}</span>', unsafe_allow_html=True)
                st.metric("Final Reconstruction Quality", f"{result_no_dp['final_quality']:.1%}")
            with col2:
                risk2 = result_with_dp["data_leakage_risk"]
                st.markdown(f'<span class="tag tag-green">With DP – Data Leakage Risk: {risk2}</span>', unsafe_allow_html=True)
                st.metric("Final Reconstruction Quality", f"{result_with_dp['final_quality']:.1%}")


# ══════════════════════════════════════════════════════
#  TAB 7 – BLOCKCHAIN
# ══════════════════════════════════════════════════════
with tab_chain:
    if "sim_results" not in st.session_state:
        st.info("Run the simulation first.")
    else:
        simulator = st.session_state["simulator"]
        tm = simulator.trust_manager

        st.markdown("### ⛓️ Blockchain Ledger")

        # Chain integrity
        is_valid = tm.verify_chain_integrity()
        if is_valid:
            st.success("✅ Blockchain integrity verified — all blocks are valid and the chain is unmodified.")
        else:
            st.error("❌ Chain integrity check FAILED — possible tampering detected!")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Blocks", len(tm.chain))
        with c2:
            st.metric("Mining Difficulty", simulator.config.mining_difficulty)
        with c3:
            st.metric("Chain Length", f"{len(tm.chain)} blocks")
        with c4:
            st.metric("Total Anomalies", len(tm.anomalies_detected))

        # ── Block viewer ──────────────────────────────────────────────────
        st.markdown("### 🔍 Block Explorer")
        max_block = len(tm.chain) - 1
        selected_block_idx = st.slider("Select Block", 0, max_block, max_block)
        block = tm.chain[selected_block_idx]

        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.markdown("**Block Header**")
            st.json({
                "index": block.index,
                "round": block.round_number,
                "nonce": block.nonce,
                "timestamp": round(block.timestamp, 3),
                "previous_hash": block.previous_hash[:24] + "…",
                "hash": block.hash[:24] + "…",
            })
        with col_r:
            st.markdown("**Trust Records in Block**")
            records_data = [
                {
                    "Client": f"C{r.client_id}",
                    "Loss": f"{r.loss:.4f}",
                    "Accuracy": f"{r.accuracy:.3f}",
                    "Grad Norm": f"{r.gradient_norm:.4f}",
                    "Trust Δ": f"{r.trust_delta:+.4f}",
                    "Flagged": "⚠️" if r.is_flagged else "✅",
                }
                for r in block.trust_records
            ]
            st.dataframe(pd.DataFrame(records_data), hide_index=True, use_container_width=True)

        # ── Hash chain visualization ──────────────────────────────────────
        st.markdown("### 🔗 Hash Chain (last 10 blocks)")
        recent_blocks = tm.chain[-10:]
        chain_data = []
        for b in recent_blocks:
            chain_data.append({
                "Block": f"#{b.index}",
                "Hash (prefix)": b.hash[:16] + "…",
                "Prev Hash (prefix)": b.previous_hash[:16] + "…",
                "Records": len(b.trust_records),
                "Nonce": b.nonce,
                "Valid": "✅" if b.is_valid() else "❌",
            })
        st.dataframe(pd.DataFrame(chain_data), hide_index=True, use_container_width=True)

        # ── Bio-similarity heatmap (last round) ──────────────────────────
        if simulator.bio_aggregator.similarity_history:
            st.markdown("### 🧬 Bio-Similarity Matrix (Last Round)")
            last_sim = simulator.bio_aggregator.similarity_history[-1]
            n = last_sim.shape[0]
            labels = [f"C{i}\n({simulator.clients[i].config.omics_type[:3]})" for i in range(n)]

            fig_heat = px.imshow(
                last_sim,
                x=labels, y=labels,
                color_continuous_scale="Viridis",
                zmin=0, zmax=1,
                template="plotly_dark",
                title="Pairwise Bio-Similarity Matrix (last aggregation round)",
                text_auto=".2f",
            )
            fig_heat.update_layout(height=500)
            st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "TA-BC-BFL Simulator · Based on: <em>Trust-Aware Blockchain-Enabled Bio-Aware Federated Learning "
    "for Privacy-Preserving Multi-Omics Analytics</em> "
    "· Abdu & Khan"
    "</div>",
    unsafe_allow_html=True,
)
