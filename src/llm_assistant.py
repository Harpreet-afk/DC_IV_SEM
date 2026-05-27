import os

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from src.rag_engine import get_rag_engine


SYSTEM_PROMPT = """You are an expert materials scientist specializing in computational 
thermodynamics, phase diagrams, and alloy design. You explain phase stability using 
rigorous thermodynamic principles (Gibbs free energy, entropy, enthalpy of mixing, 
Hume-Rothery rules). Always provide clear, scientifically accurate explanations suitable 
for an academic audience. Reference CALPHAD methodology when relevant."""


def _get_gemini_model():
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key or not HAS_GEMINI:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception:
        return None


def _call_llm(prompt, rag_context=""):
    model = _get_gemini_model()

    full_prompt = f"""{SYSTEM_PROMPT}

RELEVANT SCIENTIFIC CONTEXT:
{rag_context}

USER QUERY:
{prompt}

Provide a scientifically rigorous response."""

    if model:
        try:
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}. Using template fallback.")

    return None  


def explain_phase(T, x_zn, phase, gibbs_energy=None):
    """
    Explain why a specific phase is stable at given conditions.

    Parameters
    ----------
    T : float — Temperature (K)
    x_zn : float — Mole fraction of Zn
    phase : str — Predicted stable phase
    gibbs_energy : float, optional — Gibbs free energy (J/mol)
    """
    rag = get_rag_engine()
    context = rag.query(f"Why is {phase} stable at {T}K in Al-Zn alloy with X_Zn={x_zn}")

    prompt = (f"Explain why the phase '{phase}' is thermodynamically stable in the Al-Zn binary "
              f"alloy system at T={T}K and X(Zn)={x_zn:.3f}."
              f"{f' The Gibbs free energy is {gibbs_energy:.1f} J/mol.' if gibbs_energy else ''}"
              f" Discuss: (1) the role of each element, (2) Gibbs energy considerations, "
              f"(3) entropy vs enthalpy competition, (4) relevant Hume-Rothery rules.")

    llm_response = _call_llm(prompt, context)
    if llm_response:
        return llm_response

    # Template fallback
    x_al = 1.0 - x_zn
    responses = {
        'FCC_A1': (
            f"**Phase: FCC_A1 (Face-Centered Cubic)**\n\n"
            f"At T={T}K and X(Zn)={x_zn:.3f} (X(Al)={x_al:.3f}), the FCC_A1 phase is stable.\n\n"
            f"**Thermodynamic Reasoning:**\n"
            f"- The FCC_A1 phase represents the aluminum-rich solid solution. Aluminum's native "
            f"crystal structure is FCC, and at this composition, Al is the dominant component.\n"
            f"- Zinc has limited solubility in the Al-FCC lattice (~2.5 at% at 300K, increasing with T).\n"
            f"- The Gibbs free energy G = H - TS of FCC_A1 is minimized at this condition"
            f"{f' (G = {gibbs_energy:.1f} J/mol)' if gibbs_energy else ''}.\n\n"
            f"**Hume-Rothery Considerations:**\n"
            f"- Atomic size difference: |r_Al - r_Zn|/r_Al = {abs(143-134)/143*100:.1f}% (< 15% → favorable)\n"
            f"- Electronegativity difference: |1.61 - 1.65| = 0.04 (small → favorable)\n"
            f"- Valence electron difference: Al(3) vs Zn(2) → moderate mismatch limits solubility\n\n"
            f"**Temperature Effect:**\n"
            f"- At T={T}K, the entropy contribution (TS) {'enhances' if T > 600 else 'has limited effect on'} "
            f"solid solution stability, {'expanding' if T > 600 else 'maintaining'} the single-phase FCC region."
        ),
        'HCP_ZN': (
            f"**Phase: HCP_ZN (Hexagonal Close-Packed)**\n\n"
            f"At T={T}K and X(Zn)={x_zn:.3f}, the HCP_ZN phase is stable.\n\n"
            f"**Thermodynamic Reasoning:**\n"
            f"- HCP_ZN is the zinc-rich solid solution. Zinc's native structure is HCP.\n"
            f"- At high Zn compositions, the HCP structure has the lowest Gibbs free energy"
            f"{f' (G = {gibbs_energy:.1f} J/mol)' if gibbs_energy else ''}.\n"
            f"- Aluminum has very limited solubility in HCP-Zn (~1 at% at most temperatures).\n\n"
            f"**Phase Stability:**\n"
            f"- The Al-Zn system shows a eutectic reaction near 655K at ~95 at% Zn.\n"
            f"- Below the eutectic temperature, HCP_ZN coexists with FCC_A1 in a two-phase region."
        ),
        'LIQUID': (
            f"**Phase: LIQUID**\n\n"
            f"At T={T}K and X(Zn)={x_zn:.3f}, the liquid phase is stable.\n\n"
            f"**Thermodynamic Reasoning:**\n"
            f"- At this elevated temperature, the entropy term (TS) in G = H - TS dominates.\n"
            f"- The high configurational entropy of the liquid phase makes it thermodynamically "
            f"favorable over ordered solid phases"
            f"{f' (G = {gibbs_energy:.1f} J/mol)' if gibbs_energy else ''}.\n"
            f"- Al melts at 933K and Zn at 693K; at T={T}K, "
            f"{'both components are above their melting points' if T > 933 else 'the temperature is high enough for liquid stability at this composition'}.\n\n"
            f"**Composition Effect:**\n"
            f"- The liquidus temperature varies with composition. The minimum (eutectic) occurs near 95 at% Zn at ~655K."
        ),
    }

    base_phase = phase.split('+')[0] if '+' in phase else phase
    if base_phase in responses:
        return responses[base_phase]

    # Multi-phase or unknown
    return (
        f"**Phase: {phase}**\n\n"
        f"At T={T}K and X(Zn)={x_zn:.3f}, the system is in a multi-phase region.\n\n"
        f"This indicates the composition and temperature fall within a two-phase field "
        f"where two phases coexist in thermodynamic equilibrium. The relative amounts are "
        f"determined by the lever rule applied to the phase diagram."
        f"{f' The system Gibbs energy is {gibbs_energy:.1f} J/mol.' if gibbs_energy else ''}"
    )


def interpret_phase_diagram(phase_summary=None):
    rag = get_rag_engine()
    context = rag.query("Describe the Al-Zn binary phase diagram regions and boundaries")

    prompt = ("Provide a comprehensive interpretation of the Al-Zn binary phase diagram. "
              "Describe: (1) major phase regions, (2) phase boundaries, (3) invariant reactions "
              "(eutectic, monotectoid), (4) solubility limits, (5) the effect of temperature on phase stability.")

    llm_response = _call_llm(prompt, context)
    if llm_response:
        return llm_response

    return (
        "## Al-Zn Binary Phase Diagram Interpretation\n\n"
        "### Major Phase Regions\n"
        "1. **FCC_A1 (α-Al)**: Aluminum-rich solid solution, stable from 0 to ~2.5 at% Zn at 300K, "
        "expanding to ~67 at% Zn at the monotectoid temperature (~550K).\n"
        "2. **HCP_ZN (η-Zn)**: Zinc-rich solid solution with very limited Al solubility (<1 at%).\n"
        "3. **LIQUID**: Stable above the liquidus curve. Complete miscibility in the liquid phase.\n\n"
        "### Invariant Reactions\n"
        "- **Eutectic**: L → FCC_A1 + HCP_ZN at ~655K, ~95 at% Zn\n"
        "- **Monotectoid**: FCC_A1 → FCC_A1' + HCP_ZN at ~550K (miscibility gap in FCC phase)\n\n"
        "### Key Features\n"
        "- A **miscibility gap** exists in the FCC phase, creating a two-phase FCC+FCC' region.\n"
        "- **Retrograde solubility**: Zn solubility in FCC-Al decreases below the monotectoid temperature.\n"
        "- At high temperatures (>930K for Al-rich, >693K for Zn-rich), the system is fully liquid.\n\n"
        "### Temperature Effects\n"
        "- Increasing temperature expands single-phase regions due to the entropy contribution (TS) "
        "to the Gibbs free energy (G = H - TS).\n"
        "- The solid solubility of Zn in Al increases with temperature (up to the monotectoid)."
    )


def answer_query(question):
    rag = get_rag_engine()
    context = rag.query(question)

    llm_response = _call_llm(question, context)
    if llm_response:
        return llm_response

    # Template fallback for common questions
    q_lower = question.lower()

    if 'high temperature' in q_lower or 'temperature' in q_lower:
        return (
            "At higher temperatures in the Al-Zn system, the entropy contribution (TS) to the Gibbs "
            "free energy (G = H - TS) becomes increasingly dominant. This has several effects:\n\n"
            "1. **Expanded solid solution regions**: Higher thermal energy allows greater atomic "
            "disorder, increasing the solubility of Zn in the Al-FCC lattice.\n"
            "2. **Liquid phase stability**: Above the liquidus, the liquid phase has the lowest "
            "Gibbs energy due to its high configurational entropy.\n"
            "3. **Disappearance of ordered phases**: Intermediate compounds become unstable as "
            "the entropic driving force favors disordered solid solutions."
        )

    if 'gibbs' in q_lower or 'free energy' in q_lower:
        return (
            "Gibbs free energy (G = H - TS) is the fundamental quantity governing phase stability.\n\n"
            "- The **stable phase** at any (T, composition) is the one with the **lowest G**.\n"
            "- In two-phase regions, the system's total G lies on the **common tangent** between "
            "the G-curves of the two phases.\n"
            "- Phase boundaries correspond to points where the G-curves of two phases intersect.\n"
            "- The CALPHAD method parameterizes G for each phase using sublattice models and "
            "Redlich-Kister polynomials."
        )

    return (
        f"Regarding your question about the Al-Zn system:\n\n"
        f"The Al-Zn binary system is one of the most well-studied alloy systems in materials science. "
        f"It features a relatively simple phase diagram with FCC (Al-rich), HCP (Zn-rich), and Liquid phases.\n\n"
        f"**Retrieved Context:**\n{context}\n\n"
        f"For more detailed analysis, please provide a specific temperature and composition, "
        f"or configure the GEMINI_API_KEY for advanced AI-powered explanations."
    )
