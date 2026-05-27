import pandas as pd
import numpy as np

ELEMENT_PROPERTIES = {
    'AL': {
        'atomic_radius': 143, 'electronegativity': 1.61,
        'valence_electrons': 3, 'atomic_mass': 26.98, 'melting_point': 933.47,
    },
    'ZN': {
        'atomic_radius': 134, 'electronegativity': 1.65,
        'valence_electrons': 2, 'atomic_mass': 65.38, 'melting_point': 692.68,
    }
}


def compute_average_atomic_radius(x_al, x_zn):
    return x_al * 143 + x_zn * 134


def compute_electronegativity_diff(x_al, x_zn):
    return abs(1.61 - 1.65) * 4 * x_al * x_zn


def compute_valence_electron_concentration(x_al, x_zn):
    return x_al * 3 + x_zn * 2


def compute_atomic_size_mismatch(x_al, x_zn):
    r_avg = compute_average_atomic_radius(x_al, x_zn)
    if r_avg == 0:
        return 0.0
    delta_sq = x_al * (1 - 143 / r_avg)**2 + x_zn * (1 - 134 / r_avg)**2
    return np.sqrt(delta_sq) * 100


def compute_mixing_entropy(x_al, x_zn):
    R = 8.314
    entropy = 0.0
    for x in [x_al, x_zn]:
        if x > 1e-10:
            entropy -= x * np.log(x)
    return R * entropy


def compute_homologous_temperature(T, x_al, x_zn):
    t_m_avg = x_al * 933.47 + x_zn * 692.68
    return T / t_m_avg if t_m_avg > 0 else 0.0


def add_engineered_features(df):
    print("Computing engineered features...")
    df = df.copy()
    df['avg_atomic_radius'] = df.apply(lambda r: compute_average_atomic_radius(r['X_AL'], r['X_ZN']), axis=1)
    df['electronegativity_diff'] = df.apply(lambda r: compute_electronegativity_diff(r['X_AL'], r['X_ZN']), axis=1)
    df['valence_electron_conc'] = df.apply(lambda r: compute_valence_electron_concentration(r['X_AL'], r['X_ZN']), axis=1)
    df['atomic_size_mismatch'] = df.apply(lambda r: compute_atomic_size_mismatch(r['X_AL'], r['X_ZN']), axis=1)
    df['mixing_entropy'] = df.apply(lambda r: compute_mixing_entropy(r['X_AL'], r['X_ZN']), axis=1)
    df['homologous_temperature'] = df.apply(lambda r: compute_homologous_temperature(r['T'], r['X_AL'], r['X_ZN']), axis=1)
    print(f"[FEATURES] Added 6 engineered features. Total columns: {len(df.columns)}")
    return df


FEATURE_DESCRIPTIONS = {
    'T': 'Temperature (K)',
    'X_ZN': 'Mole fraction of Zinc',
    'X_AL': 'Mole fraction of Aluminum (= 1 - X_ZN)',
    'avg_atomic_radius': 'Composition-weighted average atomic radius (pm)',
    'electronegativity_diff': 'Weighted electronegativity difference',
    'valence_electron_conc': 'Valence Electron Concentration (VEC)',
    'atomic_size_mismatch': 'Atomic size mismatch parameter delta (%)',
    'mixing_entropy': 'Ideal configurational entropy of mixing (J/mol·K)',
    'homologous_temperature': 'Normalized temperature T/T_m_avg',
}
