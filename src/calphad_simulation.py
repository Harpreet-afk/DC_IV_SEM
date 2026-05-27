import os
import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v


COMPONENTS = ['AL', 'ZN', 'VA']          # VA = vacancy 
PHASES = ['LIQUID', 'FCC_A1', 'HCP_ZN']  # Known phases in Al-Zn system

# Grid parameters
T_MIN, T_MAX, T_STEP = 300, 1000, 10     # Temperature range (K)
X_MIN, X_MAX, X_STEP = 0.01, 0.99, 0.02  # Mole fraction of Zn

PRESSURE = 101325  # 1 atm in Pa

def load_database(tdb_path: str) -> Database:
    if not os.path.exists(tdb_path):
        raise FileNotFoundError(f"TDB file not found: {tdb_path}")
    print(f"Loading database: {tdb_path}")
    dbf = Database(tdb_path)
    print(f"Database loaded successfully.")
    return dbf


def run_single_equilibrium(dbf: Database, temperature: float, x_zn: float):
    conditions = {
        v.T: temperature,
        v.P: PRESSURE,
        v.X('ZN'): x_zn
    }
    
    try:
        eq_result = equilibrium(dbf, COMPONENTS, PHASES, conditions)
        
        # Extract phase names, fractions, and Gibbs energy
        phase_names = eq_result.Phase.values.squeeze()
        phase_fractions = eq_result.NP.values.squeeze()
        gibbs_energy = float(eq_result.GM.values.squeeze())
        
        # Find the dominant phase
        stable_phases = []
        stable_fractions = []
        
        for i, (name, frac) in enumerate(zip(phase_names, phase_fractions)):
            name_str = str(name).strip()
            if name_str and name_str != '' and not np.isnan(frac) and frac > 1e-6:
                stable_phases.append(name_str)
                stable_fractions.append(float(frac))
        
        if len(stable_phases) == 0:
            return None
        
        # Determine label
        if len(stable_phases) == 1:
            phase_label = stable_phases[0]
            phase_frac = stable_fractions[0]
        else:
            max_idx = np.argmax(stable_fractions)
            phase_label = '+'.join(stable_phases) 
            phase_frac = stable_fractions[max_idx]
        
        return {
            'T': temperature,
            'X_ZN': x_zn,
            'X_AL': 1.0 - x_zn,
            'Phase': phase_label,
            'Phase_fraction': phase_frac,
            'Gibbs_energy': gibbs_energy,
            'num_phases': len(stable_phases)
        }
        
    except Exception as e:
        print(f"  [WARN] Equilibrium failed at T={temperature}K, X_ZN={x_zn:.3f}: {e}")
        return None


def generate_dataset(tdb_path: str, output_path: str = None, 
                     t_range=None, x_range=None) -> pd.DataFrame:
    dbf = load_database(tdb_path)
    
    if t_range is None:
        t_range = (T_MIN, T_MAX, T_STEP)
    if x_range is None:
        x_range = (X_MIN, X_MAX, X_STEP)
    
    temperatures = np.arange(t_range[0], t_range[1] + 1, t_range[2])
    compositions = np.arange(x_range[0], x_range[1] + x_range[2]/2, x_range[2])
    
    total_points = len(temperatures) * len(compositions)
    print(f"Generating dataset: {len(temperatures)} temps × {len(compositions)} comps = {total_points} points")
    
    records = []
    count = 0
    
    for T in temperatures:
        for x_zn in compositions:
            count += 1
            if count % 200 == 0:
                print(f"  Progress: {count}/{total_points} ({100*count/total_points:.1f}%)")
            
            result = run_single_equilibrium(dbf, float(T), float(x_zn))
            if result is not None:
                records.append(result)
    
    df = pd.DataFrame(records)
    print(f"Dataset generated: {len(df)} valid data points out of {total_points}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
    
    return df


def generate_binplot(tdb_path: str, output_path: str = None):
    from pycalphad import binplot
    import matplotlib.pyplot as plt
    
    dbf = load_database(tdb_path)
    
    conds = {
        v.T: (300, 1000, 5),
        v.X('ZN'): (0, 1, 0.005),
        v.P: PRESSURE
    }
    
    print("Generating binary phase diagram (binplot)...")
    fig = plt.figure(figsize=(10, 8))
    axes = binplot(dbf, COMPONENTS, PHASES, conds, plot_kwargs={'fig': fig})
    
    plt.title('Al-Zn Binary Phase Diagram (CALPHAD / COST507)', fontsize=14, fontweight='bold')
    plt.xlabel('Mole Fraction Zn', fontsize=12)
    plt.ylabel('Temperature (K)', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Phase diagram saved to: {output_path}")
    
    return fig

    import sys
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdb_file = os.path.join(base_dir, 'data', 'COST507-modified.tdb')
    csv_output = os.path.join(base_dir, 'data', 'phase_diagram_dataset.csv')
    
    print("=" * 60)
    print("CALPHAD Dataset Generation — Al-Zn Binary System")
    print("=" * 60)
    
    df = generate_dataset(tdb_file, csv_output)
    print(f"\nDataset shape: {df.shape}")
    print(f"Phase distribution:\n{df['Phase'].value_counts()}")
    print(f"\nGibbs energy range: {df['Gibbs_energy'].min():.1f} to {df['Gibbs_energy'].max():.1f} J/mol")
