"""
Wizualizacja wyników analiz statystycznych
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Wczytaj dane
ranking = pd.read_csv('stat_ranking_algorithms.csv')
summary = pd.read_csv('stat_summary_metrics.csv')
comparisons = pd.read_csv('stat_comparisons_pairwise.csv')
wilcoxon = pd.read_csv('stat_comparisons_wilcoxon_holm.csv')

plot_counter = 11

# ============================================================================
# 1. RANKING ALGORYTMÓW
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ranking_sorted = ranking.sort_values('score', ascending=True)
colors = ['#2ecc71' if i == 0 else '#3498db' if i == 1 else '#e74c3c' 
          for i in range(len(ranking_sorted))]

ax.barh(ranking_sorted['algorithm'], ranking_sorted['score'], color=colors)
ax.set_xlabel('Łączny Ranking Score', fontsize=12)
ax.set_title('Ranking Algorytmów Klasteryzacji', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(ranking_sorted.iterrows()):
    ax.text(row['score'] + 0.01, i, f"{row['score']:.3f}", va='center')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/{plot_counter:02d}_ranking_algorithms.png", dpi=100, bbox_inches='tight')
plt.close()
plot_counter += 1

# ============================================================================
# 2. PORÓWNANIE ŚREDNICH METRYK (Mean Shift vs inne)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Porównanie Metryk - Mean Shift vs Pozostałe', fontsize=16, fontweight='bold')

metrics = ['mean_ARI', 'mean_NMI', 'mean_Silhouette', 'mean_Davies_Bouldin', 'mean_Calinski_Harabasz']
metric_names = ['ARI', 'NMI', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 3, idx % 3]
    
    data = ranking[[metric, 'algorithm']].sort_values(metric)
    colors_bars = ['#2ecc71' if 'Mean Shift' in algo else '#3498db' for algo in data['algorithm']]
    
    ax.barh(data['algorithm'], data[metric], color=colors_bars)
    ax.set_xlabel(name, fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (algo, val) in enumerate(zip(data['algorithm'], data[metric])):
        ax.text(val + val*0.02, i, f"{val:.3f}", va='center', fontsize=9)

# Usuń ostatni pusty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/{plot_counter:02d}_metrics_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
plot_counter += 1

# ============================================================================
# 3. STATYSTYKI ARI (średnia, min, max, std)
# ============================================================================
ari_data = summary[summary['metric'] == 'ARI'][['algorithm', 'mean', 'std', 'min', 'max']]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(ari_data))
width = 0.35

algos = ari_data['algorithm'].values
means = ari_data['mean'].values
stds = ari_data['std'].values

colors_bars = ['#2ecc71' if 'Mean Shift' in algo else '#3498db' for algo in algos]

ax.bar(x, means, width, label='Średnia', color=colors_bars, alpha=0.8)
ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, capthick=2, label='Std Dev')

ax.set_ylabel('ARI Score', fontsize=12)
ax.set_title('Stabilność Algorytmów - Metryka ARI', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(algos, rotation=15)
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/{plot_counter:02d}_ari_statistics.png", dpi=100, bbox_inches='tight')
plt.close()
plot_counter += 1

# ============================================================================
# 4. HEATMAPA PORÓWNAŃ WILCOXONA (istotność statystyczna)
# ============================================================================
wilcoxon_ari = wilcoxon[wilcoxon['metric'] == 'ARI'].copy()

# Stwórz macierz istotności
algos_all = sorted(pd.concat([wilcoxon_ari['algorithm_1'], wilcoxon_ari['algorithm_2']]).unique())
n_algos = len(algos_all)
significance_matrix = np.zeros((n_algos, n_algos))

for idx, row in wilcoxon_ari.iterrows():
    i = algos_all.index(row['algorithm_1'])
    j = algos_all.index(row['algorithm_2'])
    # 1 = istotny, 0 = nie istotny
    sig = 1 if row['significant'] == 'Yes' else 0
    significance_matrix[i, j] = sig
    significance_matrix[j, i] = sig

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    significance_matrix,
    annot=True,
    fmt='.0f',
    cmap='RdYlGn',
    cbar_kws={'label': 'Istotność (1=Tak, 0=Nie)'},
    xticklabels=algos_all,
    yticklabels=algos_all,
    vmin=0,
    vmax=1,
    ax=ax
)

ax.set_title('Test Wilcoxona z Poprawką Holma - Metryka ARI\n(Istotność Statystyczna Różnic)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/{plot_counter:02d}_wilcoxon_heatmap.png", dpi=100, bbox_inches='tight')
plt.close()
plot_counter += 1

# ============================================================================
# 5. PORÓWNANIE METRYK - BOX PLOT
# ============================================================================
ari_by_algo = summary[summary['metric'] == 'ARI'].copy()
nmi_by_algo = summary[summary['metric'] == 'NMI'].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ARI
ari_means = ari_by_algo.sort_values('mean', ascending=False)
colors_ari = ['#2ecc71' if 'Mean Shift' in algo else '#3498db' for algo in ari_means['algorithm']]
ax1.bar(ari_means['algorithm'], ari_means['mean'], color=colors_ari, alpha=0.8, label='Średnia')
ax1.errorbar(ari_means['algorithm'], ari_means['mean'], yerr=ari_means['std'], 
             fmt='none', ecolor='black', capsize=5, capthick=2)
ax1.set_ylabel('ARI', fontsize=12)
ax1.set_title('Stabilność - Metryka ARI', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=15)

# NMI
nmi_means = nmi_by_algo.sort_values('mean', ascending=False)
colors_nmi = ['#2ecc71' if 'Mean Shift' in algo else '#3498db' for algo in nmi_means['algorithm']]
ax2.bar(nmi_means['algorithm'], nmi_means['mean'], color=colors_nmi, alpha=0.8, label='Średnia')
ax2.errorbar(nmi_means['algorithm'], nmi_means['mean'], yerr=nmi_means['std'], 
             fmt='none', ecolor='black', capsize=5, capthick=2)
ax2.set_ylabel('NMI', fontsize=12)
ax2.set_title('Stabilność - Metryka NMI', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/{plot_counter:02d}_ari_nmi_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
plot_counter += 1

# ============================================================================
# 6. PODSUMOWANIE - Wszystkie metryki dla Mean Shift
# ============================================================================
meanshift_data = summary[summary['algorithm'] == 'Custom (Mean Shift)'].copy()
meanshift_data = meanshift_data.sort_values('mean', ascending=True)

# Konwertuj na numeric
meanshift_data['mean'] = pd.to_numeric(meanshift_data['mean'], errors='coerce')
meanshift_data['std'] = pd.to_numeric(meanshift_data['std'], errors='coerce')

fig, ax = plt.subplots(figsize=(10, 6))

colors_ms = ['#2ecc71' if metric in ['ARI', 'NMI', 'Silhouette', 'Calinski-Harabasz'] 
             else '#e74c3c' for metric in meanshift_data['metric']]

# Usuń NaN przed rysowaniem
valid_data = meanshift_data.dropna(subset=['mean', 'std']).reset_index(drop=True)

x_pos = np.arange(len(valid_data))
ax.bar(x_pos, valid_data['mean'].values, color=colors_ms[:len(valid_data)], alpha=0.8)
ax.errorbar(x_pos, valid_data['mean'].values, yerr=valid_data['std'].values,
            fmt='none', ecolor='black', capsize=5, capthick=2)
ax.set_ylabel('Wartość Metryki', fontsize=12)
ax.set_title('Mean Shift Clustering - Wszystkie Metryki', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(valid_data['metric'].values, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/{plot_counter:02d}_meanshift_metrics.png", dpi=100, bbox_inches='tight')
plt.close()
plot_counter += 1

print(f"\n✅ Wykresy statystyczne zapisane w folderze: {PLOTS_DIR}/")
print(f"   - 11: Ranking Algorytmów")
print(f"   - 12: Porównanie Metryk")
print(f"   - 13: Statystyki ARI")
print(f"   - 14: Heatmapa Wilcoxona")
print(f"   - 15: Porównanie ARI/NMI")
print(f"   - 16: Mean Shift - Wszystkie Metryki")
