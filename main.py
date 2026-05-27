import os
import sys
import time
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.calphad_simulation import generate_dataset
from src.feature_engineering import add_engineered_features
from src.ml_models import (train_classification, train_regression,
                           get_ml_predictions_grid, save_metrics)
from src.phase_diagram_plotter import (plot_phase_map, plot_comparison,
                                       plot_gibbs_energy_contour,
                                       plot_phase_fraction_heatmap,
                                       plot_model_comparison,
                                       plot_speed_comparison)
from src.llm_assistant import explain_phase, interpret_phase_diagram, answer_query


DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TDB_FILE = os.path.join(DATA_DIR, 'COST507-modified.tdb')
DATASET_CSV = os.path.join(DATA_DIR, 'phase_diagram_dataset.csv')
KB_JSON = os.path.join(DATA_DIR, 'knowledge_base', 'materials_science_docs.json')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')


def generate_dataset():
    print("\n" + "=" * 70)
    print("STEP 1: CALPHAD Dataset Generation")
    print("=" * 70)

    if os.path.exists(DATASET_CSV):
        print(f"Dataset already exists: {DATASET_CSV}")
        print("Loading cached dataset...")
        df = pd.read_csv(DATASET_CSV)
        print(f"Loaded {len(df)} data points.")
    else:
        if not os.path.exists(TDB_FILE):
            print(f"TDB file not found: {TDB_FILE}")
            print("Please download COST507-modified.tdb from:")
            print("  https://gist.github.com/bocklund/c4714ddbc0500c78e6fe255a763e7550")
            print(f"  and place it at: {TDB_FILE}")
            sys.exit(1)

        start = time.time()
        df = generate_dataset(TDB_FILE, DATASET_CSV)
        elapsed = time.time() - start
        print(f"\nCALPHAD generation completed in {elapsed:.1f}s")

    print(f"\nDataset shape: {df.shape}")
    print(f"Phase distribution:\n{df['Phase'].value_counts().to_string()}")
    return df


def engineer_features(df):
    print("\n" + "=" * 70)
    print("STEP 2: Feature Engineering")
    print("=" * 70)

    df = add_engineered_features(df)
    print(f"Feature columns: {list(df.columns)}")
    return df


def train_models(df):
    print("\n" + "=" * 70)
    print("STEP 3: Machine Learning Model Training")
    print("=" * 70)

    # Classification
    clf_results, clf_models, clf_scaler, clf_le = train_classification(df, MODELS_DIR)

    # Regression — Gibbs energy
    reg_gibbs_results, _, _ = train_regression(df, 'Gibbs_energy', MODELS_DIR)

    # Regression — Phase fraction
    reg_frac_results, _, _ = train_regression(df, 'Phase_fraction', MODELS_DIR)

    # Save metrics
    metrics_path = os.path.join(OUTPUTS_DIR, 'training_metrics.json')
    save_metrics(clf_results, reg_gibbs_results, reg_frac_results, metrics_path)

    return clf_results, clf_models, clf_scaler, clf_le


def visualize(df, clf_results, clf_models, clf_scaler, clf_le):
    print("\n" + "=" * 70)
    print("STEP 4: Phase Diagram Visualization")
    print("=" * 70)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # CALPHAD phase map
    plot_phase_map(df, 'Phase', 'Al-Zn Phase Diagram (CALPHAD Ground Truth)',
                   os.path.join(OUTPUTS_DIR, 'phase_diagram_calphad.png'))

    # CALPHAD vs ML comparison
    from src.ml_models import FEATURE_COLS
    import joblib
    feat_cols = joblib.load(os.path.join(MODELS_DIR, 'clf_feature_cols.pkl'))
    best_model_name = max(clf_results, key=lambda k: clf_results[k]['accuracy'])
    best_model = clf_models[best_model_name]
    ml_phases = get_ml_predictions_grid(df, best_model, clf_scaler, clf_le, feat_cols)
    plot_comparison(df, ml_phases,
                    f'CALPHAD vs {best_model_name} Phase Prediction',
                    os.path.join(OUTPUTS_DIR, 'phase_diagram_comparison.png'))

    # Gibbs energy contour
    plot_gibbs_energy_contour(df, os.path.join(OUTPUTS_DIR, 'gibbs_energy_contour.png'))

    # Phase fraction heatmap
    plot_phase_fraction_heatmap(df, os.path.join(OUTPUTS_DIR, 'phase_fraction_heatmap.png'))

    # Model comparison
    plot_model_comparison(clf_results, os.path.join(OUTPUTS_DIR, 'model_comparison.png'))

    # Speed comparison
    plot_speed_comparison(clf_results, save_path=os.path.join(OUTPUTS_DIR, 'speed_comparison.png'))

    print(f"\nAll visualizations saved to: {OUTPUTS_DIR}")


def llm_demo():
    print("\n" + "=" * 70)
    print("  STEP 5: LLM Scientific Assistant Demo")
    print("=" * 70)

    # Initialize RAG
    try:
        from src.rag_engine import get_rag_engine
        rag = get_rag_engine(os.path.join(DATA_DIR, 'chroma_db'))
        if rag.initialized and os.path.exists(KB_JSON):
            rag.populate_from_json(KB_JSON)
    except Exception as e:
        print(f"[MAIN] RAG initialization note: {e}")

    print("\nPhase Explanation Demo")
    explanation = explain_phase(700, 0.3, 'FCC_A1', -35000)
    print(explanation[:500] + "..." if len(explanation) > 500 else explanation)

    print("\nPhase Diagram Interpretation Demo")
    interpretation = interpret_phase_diagram()
    print(interpretation[:500] + "..." if len(interpretation) > 500 else interpretation)

    print("\nQ&A Demo")
    answer = answer_query("What happens to phase stability in Al-Zn alloys at high temperature?")
    print(answer[:500] + "..." if len(answer) > 500 else answer)


def main():
    """Run the complete pipeline."""
    print("+" + "=" * 68 + "+")
    print("AI-Accelerated Phase Diagram Prediction")
    print("Binary Alloy System: Al-Zn")
    print("Using COST507 Thermodynamic Database")
    print("+" + "=" * 68 + "+")

    total_start = time.time()

    df = generate_dataset()

    df = engineer_features(df)

    clf_results, clf_models, clf_scaler, clf_le = train_models(df)

    visualize(df, clf_results, clf_models, clf_scaler, clf_le)

    llm_demo()

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUTS_DIR}")
    print(f"Models saved to:  {MODELS_DIR}")
    print(f"\nTo launch the interactive web app:")
    print(f"  streamlit run app.py")

if __name__ == '__main__':
    main()
