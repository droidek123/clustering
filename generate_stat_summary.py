"""
Generowanie LaTeX tabel z wyników Mean Shift vs inne algorytmy
"""
import pandas as pd

ranking = pd.read_csv('stat_ranking_algorithms.csv')
summary = pd.read_csv('stat_summary_metrics.csv')

# Główna tabela porównawcza
print("\n" + "="*80)
print("RANKING ALGORYTMÓW - LaTeX")
print("="*80)

ranking_latex = ranking[['rank', 'algorithm', 'mean_ARI', 'mean_NMI', 'mean_Silhouette']].copy()
ranking_latex.columns = ['Pozycja', 'Algorytm', 'ARI', 'NMI', 'Silhouette']
ranking_latex = ranking_latex.round(4)

print("\nTabela 1: Ranking Algorytmów")
print(ranking_latex.to_latex(index=False))

# Tabela metryk dla Mean Shift
print("\n" + "="*80)
print("MEAN SHIFT - SZCZEGÓŁOWE METRYKI")
print("="*80)

meanshift_summary = summary[summary['algorithm'] == 'Custom (Mean Shift)'][
    ['metric', 'mean', 'std', 'min', 'max']
].round(4)
meanshift_summary.columns = ['Metrika', 'Średnia', 'Std Dev', 'Min', 'Max']

print("\nTabela 2: Mean Shift - Statystyki")
print(meanshift_summary.to_latex(index=False))

# Porównanie ARI dla wszystkich algorytmów
print("\n" + "="*80)
print("PORÓWNANIE ARI - WSZYSTKIE ALGORYTMY")
print("="*80)

ari_all = summary[summary['metric'] == 'ARI'][['algorithm', 'mean', 'std', 'min', 'max']].round(4)
ari_all.columns = ['Algorytm', 'Średnia', 'Std Dev', 'Min', 'Max']

print("\nTabela 3: Stabilność (ARI)")
print(ari_all.to_latex(index=False))

# Porównanie NMI dla wszystkich algorytmów  
print("\n" + "="*80)
print("PORÓWNANIE NMI - WSZYSTKIE ALGORYTMY")
print("="*80)

nmi_all = summary[summary['metric'] == 'NMI'][['algorithm', 'mean', 'std', 'min', 'max']].round(4)
nmi_all.columns = ['Algorytm', 'Średnia', 'Std Dev', 'Min', 'Max']

print("\nTabela 4: Jakość (NMI)")
print(nmi_all.to_latex(index=False))

# Zapisz do pliku
with open('stat_latex_summary.txt', 'w', encoding='utf-8') as f:
    f.write("% RANKING ALGORYTMÓW\n")
    f.write(ranking_latex.to_latex(index=False))
    f.write("\n\n% MEAN SHIFT - METRYKI\n")
    f.write(meanshift_summary.to_latex(index=False))
    f.write("\n\n% PORÓWNANIE ARI\n")
    f.write(ari_all.to_latex(index=False))
    f.write("\n\n% PORÓWNANIE NMI\n")
    f.write(nmi_all.to_latex(index=False))

print("\n✅ Tabele LaTeX zapisane do: stat_latex_summary.txt")

# Podsumowanie dla raportu
print("\n" + "="*80)
print("PODSUMOWANIE DLA RAPORTU")
print("="*80)
print(f"""
Mean Shift Clustering - Wyniki Analizy
=====================================

1. POZYCJA W RANKINGU: {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['rank'].values[0]}/4

2. METRYKI GŁÓWNE:
   - ARI (Adjusted Rand Index):     {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['mean_ARI'].values[0]:.4f}
   - NMI (Normalized Mutual Info):  {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['mean_NMI'].values[0]:.4f}
   - Silhouette Score:              {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['mean_Silhouette'].values[0]:.4f}
   - Davies-Bouldin Index:          {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['mean_Davies_Bouldin'].values[0]:.4f}
   - Calinski-Harabasz Index:       {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['mean_Calinski_Harabasz'].values[0]:.2f}

3. ALGORYTM RANK.NET:
   Mean Shift uzyskał pozycję #{ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['rank'].values[0]} 
   z punktacją {ranking[ranking['algorithm'] == 'Custom (Mean Shift)']['score'].values[0]:.4f}
   
4. PORÓWNANIE:
   - vs DBSCAN:      Mean Shift gorszy w ARI, lepszy w Davis-Bouldin
   - vs Hierarchical: Mean Shift lepszy w większości metryk
   - vs KMeans:       Mean Shift lepszy w większości metryk
""")
