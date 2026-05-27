# AI-Accelerated Phase Diagram Prediction with LLM-Based Scientific Assistant

## Binary Alloy System: Al–Zn

This project presents an integrated Materials Informatics Pipeline combining CALPHAD thermodynamic simulations, Machine Learning models, and a Large Language Model (LLM)-based scientific assistant for rapid prediction and interpretation of phase stability in binary alloy systems.

The system uses pycalphad to generate thermodynamic equilibrium datasets from TDB databases, which are then used to train ML models capable of predicting stable phases, phase fractions, and Gibbs free energy significantly faster than traditional CALPHAD calculations.

An LLM-powered assistant is integrated to:

Explain phase stability using thermodynamic reasoning
Interpret phase diagrams
Retrieve and summarize scientific insights from materials science literature using Retrieval-Augmented Generation (RAG)

Main Pipeline Runner

Runs the complete AI-Accelerated Phase Diagram Prediction pipeline:
1. Generate CALPHAD dataset (or load existing)
2. Engineer features
3. Train ML models
4. Generate phase diagram visualizations
5. Demo LLM assistant

## Pipeline Structure
User Input (Temperature + Composition)
                    │
                    ▼
        ┌────────────────────┐
        │    PyCALPHAD       │
        │ Thermodynamic Calc │
        └────────────────────┘
                    │
                    ▼
        Generated Thermodynamic Dataset
                    │
                    ▼
        ┌────────────────────┐
        │  Feature Engineering│
        └────────────────────┘
                    │
                    ▼
        ┌────────────────────┐
        │   ML Models        │
        │ RF / XGBoost / MLP │
        └────────────────────┘
                    │
                    ▼
      Phase / Gibbs Energy Prediction
                    │
                    ▼
        ┌────────────────────┐
        │   LLM Assistant    │
        └────────────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼                              ▼
Phase Explanation         Literature-Grounded Reasoning

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

## LLM Assistant Features
The integrated LLM assistant can:
Explain stable phase formation using Gibbs free energy trends
Interpret binary phase diagrams
Provide thermodynamic reasoning for alloy behavior
Retrieve literature-supported explanations using RAG pipelines

## Technologies Used
**Core Technologies**
Thermodynamic Simulation
   1. pycalphad
   2. TDB databases
**Machine Learning**
   1. Scikit-learn
   2. XGBoost
   3. PyTorch / TensorFlow
**Data Processing**
   1. NumPy
   2. Pandas
   3. Matplotlib
   4. Plotly
**LLM + RAG**
   1. LangChain
   2. FAISS
   3. OpenAI / Llama Models
**Deployment / Interface**
   Streamlit

## References
PyCALPHAD Documentation
CALPHAD Methodology
Materials Informatics Literature
Scientific LLM and RAG Research Papers

