import streamlit as st
import os
import json
import yaml
import tempfile
import pandas as pd
from src.orchestrator import Orchestrator
from src.core.llm_client import LLMClient
import shutil
from datetime import datetime

st.set_page_config(
    page_title="Prompt Optimizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-card.cost {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 12px;
        opacity: 0.8;
    }
    .stTextArea textarea {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 13px;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("⚠️ Set GEMINI_API_KEY in environment variables")
    else:
        st.success("✓ API Key loaded")

    model_name = st.selectbox(
        "Select Model",
        [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-exp",
            "gemini-flash-latest",
            "gemini-1.5-flash",
        ],
        index=0,
    )

    st.divider()

    st.subheader("🔧 Optimization Settings")
    max_iters = st.number_input("Max Iterations", min_value=1, max_value=10, value=3)

    st.subheader("🚀 Multi-Variant Mode")
    enable_variants = st.toggle("Enable Multi-Variant Optimization", value=True)
    num_variants = (
        st.number_input("Number of Variants", min_value=2, max_value=5, value=3)
        if enable_variants
        else 3
    )

    st.subheader("📊 Scoring Weights")
    exact_weight = st.slider("Exact Match", 0.0, 1.0, 0.30)
    semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.30)
    partial_weight = st.slider("Partial Credit", 0.0, 1.0, 0.25)
    confidence_weight = st.slider("Confidence", 0.0, 1.0, 0.15)

    st.divider()

    st.subheader("🛡️ Anti-Overfitting")
    k_folds = st.number_input("CV K-Folds", min_value=2, max_value=10, value=5)
    gen_gap_limit = st.slider("Generalization Gap Limit", 0.05, 0.30, 0.15)
    gini_threshold = st.slider("Gini Threshold", 0.1, 0.5, 0.3)

st.title("✨ Prompt Optimizer")
st.markdown(
    "**Multi-dimensional evaluation • Anti-overfitting • Real-world reliability**"
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Define Task")
    task_intent = st.text_area(
        "Task Intent",
        height=200,
        placeholder="E.g., Extract tender information from newspaper clippings...",
        value='Extract the following fields from the images:\n{\n  "tenders": [{\n    "procurement_type": null,\n    "amount": null,\n    "authority_name": null,\n    "bid_deadline": null\n  }]\n}',
    )

    base_prompt = st.text_area(
        "Base Context (Optional)",
        height=150,
        placeholder="You are an expert analyst...",
        value="You are an expert tender analyst. Be precise with monetary values. Handle various date formats. If a field is not present, return null.",
    )

with col2:
    st.subheader("2. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload 5-10 diverse examples for best results",
    )

    ground_truth_input = st.text_area(
        "Ground Truth JSON",
        height=300,
        placeholder='[\n  {"tenders": [{"amount": "80000", ...}]}, \n  {"tenders": [{"amount": "2496537", ...}]}\n]',
        help="Must match number of uploaded images",
    )

if st.button(
    "🚀 Start Optimization",
    type="primary",
    disabled=not (api_key and task_intent and uploaded_files and ground_truth_input),
):
    try:
        ground_truth_list = json.loads(ground_truth_input)
        if len(ground_truth_list) != len(uploaded_files):
            st.error(
                f"Mismatch: {len(uploaded_files)} images but {len(ground_truth_list)} ground truth entries."
            )
            st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in Ground Truth: {e}")
        st.stop()

    progress_container = st.container()
    metrics_container = st.container()

    with progress_container:
        st.subheader("📈 Optimization Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.expander("Real-time Logs", expanded=True)

    demo_data = []
    temp_dir = tempfile.mkdtemp()

    for i, file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        demo_data.append({"input": file_path, "output": ground_truth_list[i]})
        log_area.write(f"Loaded: {file.name}")

    try:
        orchestrator = Orchestrator(
            llm_provider="gemini", api_key=api_key, model_name=model_name
        )

        status_text.text("Initializing optimization...")
        progress_bar.progress(10)

        if enable_variants:
            status_text.text("Running multi-variant optimization...")
            result = orchestrator.optimize_with_variants(
                task_intent,
                demo_data,
                max_iterations=max_iters,
                base_prompt_context=base_prompt,
                num_variants=num_variants,
            )
        else:
            status_text.text("Running standard optimization...")
            result = orchestrator.optimize_prompt(
                task_intent,
                demo_data,
                max_iterations=max_iters,
                base_prompt_context=base_prompt,
            )

        progress_bar.progress(100)
        status_text.text("✅ Optimization Complete!")

        shutil.rmtree(temp_dir)

        st.success("🎉 Optimization Complete!")

        st.markdown("---")
        st.subheader("💰 Cost Summary")

        total_cost = orchestrator.llm.total_cost
        total_inr = total_cost * 87.0
        input_tokens = orchestrator.llm.total_input_tokens
        output_tokens = orchestrator.llm.total_output_tokens

        col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)

        with col_cost1:
            st.markdown(
                f"""
            <div class="metric-card cost">
                <div class="metric-value">${total_cost:.4f}</div>
                <div class="metric-label">Total Cost (USD)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_cost2:
            st.markdown(
                f"""
            <div class="metric-card cost">
                <div class="metric-value">₹{total_inr:.2f}</div>
                <div class="metric-label">Total Cost (INR)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_cost3:
            st.markdown(
                f"""
            <div class="metric-card cost">
                <div class="metric-value">{input_tokens:,}</div>
                <div class="metric-label">Input Tokens</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_cost4:
            st.markdown(
                f"""
            <div class="metric-card cost">
                <div class="metric-value">{output_tokens:,}</div>
                <div class="metric-label">Output Tokens</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with st.expander("📊 Cost Breakdown"):
            st.markdown("### Cost Breakdown by Call")
            st.markdown(f"""
            - **Input Cost**: ${input_tokens / 1_000_000 * 0.075:.6f} ({input_tokens:,} tokens @ $0.075/M)
            - **Output Cost**: ${output_tokens / 1_000_000 * 0.30:.6f} ({output_tokens:,} tokens @ $0.30/M)
            - **Total**: ${total_cost:.6f}
            """)

            est_production_cost = total_cost * 10
            st.info(
                f"📌 **Estimated Production Cost**: ~${est_production_cost:.2f} for 10x the inference calls"
            )

        st.markdown("---")
        st.subheader("🎯 Optimization Results")

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{result.get("best_score", 0):.1%}</div>
                <div class="metric-label">Best Score</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_b:
            risk = result.get("overfitting_risk", "unknown")
            risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{risk_color} {risk.upper()}</div>
                <div class="metric-label">Overfitting Risk</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_c:
            diverse = result.get("diversity_check", True)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{"✅" if diverse else "❌"} {"PASS" if diverse else "FAIL"}</div>
                <div class="metric-label">Diversity Check</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_d:
            edge_cases = result.get("edge_cases", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{edge_cases}</div>
                <div class="metric-label">Edge Cases</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if "score_history" in result and result["score_history"]:
            st.subheader("📉 Score Progression")
            score_df = pd.DataFrame(
                {
                    "Iteration": range(1, len(result["score_history"]) + 1),
                    "Score": result["score_history"],
                }
            )
            st.line_chart(score_df.set_index("Iteration"))

        if "variant_scores" in result:
            st.subheader("🔀 Variant Comparison")
            var_df = pd.DataFrame(
                [
                    {"Variant": k, "Score": v}
                    for k, v in result["variant_scores"].items()
                ]
            )
            st.bar_chart(var_df.set_index("Variant"))

        st.subheader("📝 Frozen Prompt")
        frozen_prompt = result.get("frozen_prompt", "Error: No prompt returned.")

        tab1, tab2, tab3 = st.tabs(["YAML View", "Formatted", "Copy"])
        with tab1:
            st.code(frozen_prompt, language="yaml")
        with tab2:
            try:
                prompt_dict = yaml.safe_load(frozen_prompt)
                for section, content in prompt_dict.items():
                    with st.expander(f"**{section.upper()}**"):
                        st.markdown(content)
            except:
                st.warning("Could not parse as YAML")
                st.code(frozen_prompt)
        with tab3:
            st.code(frozen_prompt, language="yaml")
            st.button(
                "📋 Copy to Clipboard",
                on_click=lambda: st.write_to_clipboard(frozen_prompt),
            )

        with st.expander("📋 Advanced Diagnostics"):
            st.json(
                {
                    "score_history": result.get("score_history", []),
                    "overfitting_risk": result.get("overfitting_risk"),
                    "diversity_check": result.get("diversity_check"),
                    "edge_cases": result.get("edge_cases"),
                    "generalization_gap": result.get("generalization_gap"),
                }
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback

        st.code(traceback.format_exc())
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666;">
    <b>Production-Ready Prompt Optimizer</b><br>
    Multi-dimensional scoring • Anti-overfitting • Real-world reliability
</div>
""",
    unsafe_allow_html=True,
)
