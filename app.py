import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import add_engineered_features, FEATURE_DESCRIPTIONS
from src.phase_diagram_plotter import (plot_phase_map, plot_gibbs_energy_contour,
                                       plot_phase_fraction_heatmap, plot_model_comparison)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TDB_FILE = os.path.join(DATA_DIR, 'COST507-modified.tdb')
DATASET_CSV = os.path.join(DATA_DIR, 'phase_diagram_dataset.csv')
KB_JSON = os.path.join(DATA_DIR, 'knowledge_base', 'materials_science_docs.json')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
METRICS_FILE = os.path.join(OUTPUTS_DIR, 'training_metrics.json')

st.set_page_config(
    page_title="AI Phase Diagram Predictor — Al-Zn",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin-bottom: 0.5rem; }
    .main-header p { color: #bbdefb; font-size: 1.1rem; }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1.2rem; border-radius: 10px; text-align: center;
        border: 1px solid #90caf9;
    }
    .metric-card h3 { color: #1565c0; margin: 0; font-size: 1.8rem; }
    .metric-card p { color: #37474f; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e3f2fd; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1> AI-Accelerated Phase Diagram Prediction</h1>
        <p>Binary Alloy System: Al–Zn | CALPHAD + Machine Learning + LLM Assistant</p>
    </div>
    """, unsafe_allow_html=True)


def load_dataset():
    if os.path.exists(DATASET_CSV):
        df = pd.read_csv(DATASET_CSV)
        if 'avg_atomic_radius' not in df.columns:
            df = add_engineered_features(df)
        return df
    return None


def tab_dataset():
    st.header("Phase Equilibrium Dataset")

    df = load_dataset()
    if df is None:
        st.warning("Dataset not generated yet. Run `python main.py` first to generate the CALPHAD dataset.")
        st.code("python main.py", language="bash")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(df):,}</h3><p>Data Points</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{df["Phase"].nunique()}</h3><p>Unique Phases</p></div>', unsafe_allow_html=True)
    with col3:
        t_range = f'{df["T"].min():.0f}–{df["T"].max():.0f}'
        st.markdown(f'<div class="metric-card"><h3>{t_range}</h3><p>Temp Range (K)</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{len(df.columns)}</h3><p>Features</p></div>', unsafe_allow_html=True)

    st.subheader("Phase Distribution")
    phase_counts = df['Phase'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    phase_counts.plot(kind='bar', ax=ax, color=['#2196F3', '#4CAF50', '#F44336', '#FF9800'][:len(phase_counts)])
    ax.set_ylabel('Count')
    ax.set_title('Phase Distribution in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Full Dataset (CSV)", csv, "phase_diagram_dataset.csv", "text/csv")


def tab_phase_diagrams():
    st.header("Phase Diagrams")

    df = load_dataset()
    if df is None:
        st.warning("Dataset not generated yet. Run `python main.py` first.")
        return

    diagram_type = st.selectbox("Select Diagram Type", [
        "Phase Map (CALPHAD)", "Gibbs Energy Contour", "Phase Fraction Heatmap"
    ])

    if diagram_type == "Phase Map (CALPHAD)":
        fig = plot_phase_map(df, 'Phase', 'Al-Zn Phase Diagram (CALPHAD)')
        st.pyplot(fig)
    elif diagram_type == "Gibbs Energy Contour":
        fig = plot_gibbs_energy_contour(df)
        st.pyplot(fig)
    elif diagram_type == "Phase Fraction Heatmap":
        fig = plot_phase_fraction_heatmap(df)
        st.pyplot(fig)

    comparison_img = os.path.join(OUTPUTS_DIR, 'phase_diagram_comparison.png')
    if os.path.exists(comparison_img):
        st.subheader("CALPHAD vs ML Comparison")
        st.image(comparison_img, use_container_width=True)


def tab_ml_models():
    st.header("Machine Learning Models")

    if not os.path.exists(METRICS_FILE):
        st.warning("Models not trained yet. Run `python main.py` first.")
        return

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    st.subheader("Classification Results — Phase Prediction")
    clf = metrics.get('classification', {})
    if clf:
        cols = st.columns(len(clf))
        for col, (name, res) in zip(cols, clf.items()):
            with col:
                st.metric(f"{name}", f"{res['accuracy']*100:.2f}%", f"F1: {res['f1_score']:.4f}")
                st.caption(f"Train: {res['train_time']}s | Pred: {res['pred_time']}s")

        fig = plot_model_comparison(clf)
        st.pyplot(fig)

    st.subheader("Regression Results — Gibbs Energy")
    reg_g = metrics.get('regression_gibbs_energy', {})
    if reg_g:
        cols = st.columns(len(reg_g))
        for col, (name, res) in zip(cols, reg_g.items()):
            with col:
                st.metric(f"{name}", f"R² = {res['r2_score']:.4f}")
                st.caption(f"MAE: {res['mae']:.1f} | RMSE: {res['rmse']:.1f}")

    st.subheader("Regression Results — Phase Fraction")
    reg_f = metrics.get('regression_phase_fraction', {})
    if reg_f:
        cols = st.columns(len(reg_f))
        for col, (name, res) in zip(cols, reg_f.items()):
            with col:
                st.metric(f"{name}", f"R² = {res['r2_score']:.4f}")
                st.caption(f"MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f}")

    # Speed comparison
    speed_img = os.path.join(OUTPUTS_DIR, 'speed_comparison.png')
    if os.path.exists(speed_img):
        st.subheader("Speed Comparison: CALPHAD vs ML")
        st.image(speed_img, use_container_width=True)


def tab_prediction():
    st.header("Interactive Phase Prediction")

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (K)", 300, 1000, 600, step=10)
    with col2:
        x_zn = st.slider("Mole Fraction Zn (X_ZN)", 0.01, 0.99, 0.30, step=0.01)

    x_al = 1.0 - x_zn
    st.info(f"**Conditions:** T = {temperature}K | X(Zn) = {x_zn:.2f} | X(Al) = {x_al:.2f}")

    if st.button("Predict Phase", type="primary"):
        try:
            from src.ml_models import predict_phase
            predictions = predict_phase(temperature, x_zn, MODELS_DIR)

            st.subheader("ML Predictions")
            for model_name, phase in predictions.items():
                st.success(f"**{model_name}**: {phase}")

            # Show LLM explanation
            st.subheader("AI Reasoning")
            from src.llm_assistant import explain_phase
            dominant_phase = list(predictions.values())[0] if predictions else "Unknown"
            explanation = explain_phase(temperature, x_zn, dominant_phase)
            st.markdown(explanation)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure you've run `python main.py` to train the models first.")


def tab_assistant():
    st.header("Materials-AI Assistant")
    st.caption("Ask questions about Al-Zn phase diagrams, thermodynamics, and alloy behavior.")

    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Interpret Phase Diagram"):
            from src.llm_assistant import interpret_phase_diagram
            result = interpret_phase_diagram()
            st.markdown(result)
    with col2:
        if st.button("High Temperature Effects"):
            from src.llm_assistant import answer_query
            result = answer_query("What happens to phase stability in Al-Zn alloys at high temperature?")
            st.markdown(result)

    st.subheader("Ask a Question")
    user_question = st.text_area("Enter your materials science question:",
                                  placeholder="e.g., Why does FCC dissolve more Zn at higher temperatures?")
    if st.button("Get Answer", type="primary") and user_question:
        from src.llm_assistant import answer_query
        answer = answer_query(user_question)
        st.markdown(answer)


# Main App 
def main():
    render_header()

    # Sidebar
    with st.sidebar:
        st.image("image.png", width=80)
        st.title("Navigation")
        st.markdown("---")
        st.markdown("**System:** Al-Zn Binary Alloy")
        st.markdown("**Database:** COST507")
        st.markdown("**Models:** RF, XGBoost, MLP")
        st.markdown("---")

        if os.path.exists(DATASET_CSV):
            st.success("Dataset generated")
        else:
            st.warning("Run `python main.py`")

        if os.path.exists(MODELS_DIR) and any(f.endswith('.pkl') for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))):
            st.success("Models trained")
        else:
            st.warning("Models not trained")

        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            st.success("LLM API connected")
        else:
            st.info("Using template responses")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset", "Phase Diagrams", "ML Models",
        "Prediction", "AI Assistant"
    ])

    with tab1:
        tab_dataset()
    with tab2:
        tab_phase_diagrams()
    with tab3:
        tab_ml_models()
    with tab4:
        tab_prediction()
    with tab5:
        tab_assistant()

if __name__ == '__main__':
    main()
