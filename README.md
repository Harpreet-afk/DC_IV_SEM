# AI-Accelerated Phase Diagram Prediction with LLM-Based Scientific Assistant

## Binary Alloy System: Al–Zn

An integrated AI-driven materials informatics pipeline combining CALPHAD thermodynamic simulations, machine learning models, and an LLM-based scientific assistant to predict, analyze, and explain phase stability in the Al–Zn binary alloy system.

Main Pipeline Runner

Runs the complete AI-Accelerated Phase Diagram Prediction pipeline:
1. Generate CALPHAD dataset (or load existing)
2. Engineer features
3. Train ML models
4. Generate phase diagram visualizations
5. Demo LLM assistant


## Project Structure
DC_IV_SEM/
├── data/
│   ├── COST507-modified.tdb              # Thermodynamic database
│   ├── phase_diagram_dataset.csv         # Generated dataset (auto-created)
│   └── knowledge_base/
│       └── materials_science_docs.json   # RAG knowledge base
├── src/
│   ├── calphad_simulation.py             # PyCALPHAD dataset generation
│   ├── feature_engineering.py            # Physical property features
│   ├── ml_models.py                      # ML training & prediction
│   ├── phase_diagram_plotter.py          # Visualization module
│   ├── rag_engine.py                     # RAG with ChromaDB
│   └── llm_assistant.py                  # LLM scientific assistant
├── models/                               # Saved trained models (auto-created)
├── outputs/                              # Generated plots (auto-created)
├── app.py                                # Streamlit web app
├── main.py                               # CLI pipeline runner
├── requirements.txt                      # Dependencies
└── README.md                             # This file

### Run Full Pipeline (CLI)

This will:
1. Generate CALPHAD equilibrium dataset (~3,500 points)
2. Compute engineered features
3. Train ML models (RF, XGBoost, MLP)
4. Generate phase diagram visualizations
5. Demo the LLM assistant

### Launch Web Interface

## Dataset

The dataset is generated for the Al-Zn binary system using PyCALPHAD:

| Column | Description |
|--------|-------------|
| T | Temperature (K) |
| X_ZN | Mole fraction of Zinc |
| X_AL | Mole fraction of Aluminum |
| Phase | Stable phase label |
| Phase_fraction | Dominant phase fraction |
| Gibbs_energy | Molar Gibbs free energy (J/mol) |
| avg_atomic_radius | Weighted average atomic radius (pm) |
| electronegativity_diff | Electronegativity difference |
| valence_electron_conc | Valence Electron Concentration |
| atomic_size_mismatch | Size mismatch parameter δ (%) |
| mixing_entropy | Ideal entropy of mixing (J/mol·K) |
| homologous_temperature | Normalized temperature T/T_m |

## Machine Learning Tasks

### Classification
Predict stable phase: FCC_A1, HCP_ZN, LIQUID, or multi-phase

### Regression
Predict: Gibbs free energy (J/mol) and Phase fraction

### Speed Benchmark
ML prediction time << CALPHAD simulation time

## LLM Assistant Functions

1. Phase Explanation — Why a phase is stable at given T, X
2. Diagram Interpretation — Describe regions, boundaries, trends
3. RAG Q&A — Answer questions grounded in scientific literature

