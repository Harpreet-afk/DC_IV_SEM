import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


PHASE_COLORS = {
    'FCC_A1': '#2196F3',
    'HCP_ZN': '#4CAF50',
    'LIQUID': '#F44336',
}

def _get_color(phase_name):
    if phase_name in PHASE_COLORS:
        return PHASE_COLORS[phase_name]
    if '+' in str(phase_name):
        return '#FF9800'  
    return '#9E9E9E'  


def plot_phase_map(df, phase_col='Phase', title='Phase Diagram', save_path=None):
    # Plot T vs X(ZN) phase map colored by stable phase.
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_phases = sorted(df[phase_col].unique())
    for phase in unique_phases:
        mask = df[phase_col] == phase
        color = _get_color(phase)
        ax.scatter(df.loc[mask, 'X_ZN'], df.loc[mask, 'T'],
                   c=color, s=8, label=phase, alpha=0.7, edgecolors='none')

    ax.set_xlabel('Mole Fraction Zn (X_ZN)', fontsize=13)
    ax.set_ylabel('Temperature (K)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_comparison(df, ml_phases, title='CALPHAD vs ML Phase Diagram', save_path=None):
    # comparison of CALPHAD and ML phase diagrams
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, phases, subtitle in zip(axes, [df['Phase'], ml_phases],
                                     ['CALPHAD (Ground Truth)', 'ML Prediction']):
        unique_phases = sorted(set(phases))
        for phase in unique_phases:
            mask = phases == phase
            color = _get_color(phase)
            ax.scatter(df.loc[mask, 'X_ZN'], df.loc[mask, 'T'],
                       c=color, s=6, label=phase, alpha=0.7, edgecolors='none')
        ax.set_xlabel('Mole Fraction Zn', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title(subtitle, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, markerscale=3)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_gibbs_energy_contour(df, save_path=None):
    # Contour plot of Gibbs free energy over T-composition space
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot = df.pivot_table(values='Gibbs_energy', index='T', columns='X_ZN', aggfunc='mean')
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values

    contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Gibbs Free Energy (J/mol)', fontsize=12)
    ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    ax.set_xlabel('Mole Fraction Zn', fontsize=13)
    ax.set_ylabel('Temperature (K)', fontsize=13)
    ax.set_title('Gibbs Free Energy Landscape — Al-Zn System', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_phase_fraction_heatmap(df, save_path=None):
    # Heatmap of dominant phase fraction over T-composition space
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot = df.pivot_table(values='Phase_fraction', index='T', columns='X_ZN', aggfunc='mean')
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Phase Fraction'})

    ax.set_xlabel('Mole Fraction Zn', fontsize=13)
    ax.set_ylabel('Temperature (K)', fontsize=13)
    ax.set_title('Phase Fraction Heatmap — Al-Zn System', fontsize=15, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_model_comparison(clf_results, save_path=None):
    # Bar chart comparing model accuracy and F1 scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = list(clf_results.keys())
    accuracies = [clf_results[m]['accuracy'] for m in models]
    f1_scores = [clf_results[m]['f1_score'] for m in models]

    colors = ['#2196F3', '#4CAF50', '#FF9800']

    axes[0].bar(models, accuracies, color=colors[:len(models)], edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Classification Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0.8, 1.0)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

    axes[1].bar(models, f1_scores, color=colors[:len(models)], edgecolor='white', linewidth=1.5)
    axes[1].set_ylabel('F1 Score (Weighted)', fontsize=12)
    axes[1].set_title('Classification F1 Score', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0.8, 1.0)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_speed_comparison(clf_results, calphad_time_per_point=0.05, n_points=3500, save_path=None):
    # Bar chart comparing CALPHAD vs ML prediction speeds
    fig, ax = plt.subplots(figsize=(10, 6))

    calphad_total = calphad_time_per_point * n_points
    labels = ['CALPHAD\n(Simulation)']
    times = [calphad_total]
    colors = ['#F44336']

    for name, res in clf_results.items():
        labels.append(f'{name}\n(ML)')
        times.append(res['pred_time'])
        colors.append('#4CAF50')

    bars = ax.bar(labels, times, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Total Prediction Time (seconds)', fontsize=12)
    ax.set_title('Speed Comparison: CALPHAD vs ML Models', fontsize=15, fontweight='bold')
    ax.set_yscale('log')

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                f'{t:.4f}s', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig
