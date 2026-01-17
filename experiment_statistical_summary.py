"""
Analiza statystyczna wyników eksperymentów — porównanie algorytmów.

Zbiera wyniki metryk z każdego algorytmu i tworzy:
1. Podsumowania statystyczne (średnia, mediana, std, percentyle)
2. Porównania par algorytmów (test Mann-Whitney U)
3. Zapisuje do CSV
"""
from city_data_loader import load_city_dataset
from stability_analysis import stability_with_metrics
from statistical_analysis import summarize_metrics, compare_algorithms, compare_algorithms_wilcoxon_holm, save_results, rank_algorithms
import pandas as pd

from clustering_algorithms import (
    kmeans_clustering,
    hierarchical_clustering,
    dbscan_clustering
)
from custom_clustering import custom_clustering


def main():
    X, _, _ = load_city_dataset("data/polish_cities_2020.csv")

    algorithms = {
        "KMeans": (kmeans_clustering, {"n_clusters": 3}),
        "Hierarchical": (hierarchical_clustering, {"n_clusters": 3}),
        "DBSCAN": (dbscan_clustering, {"eps": 2.32, "min_samples": 2}),
        "Custom": (custom_clustering, {"n_clusters": 4}),
    }

    print("\n=== ANALIZA STATYSTYCZNA WYNIKÓW ===\n")

    # Zbieranie wyników dla każdego algorytmu
    all_results = {}
    summary_dfs = []
    ari_pairs = []
    nmi_pairs = []
    silhouette_pairs = []

    for name, (fn, params) in algorithms.items():
        print(f"\nAnalizuję: {name}")
        result = stability_with_metrics(
            clustering_fn=fn,
            X=X,
            fn_params=params,
            algorithm_name=name,
            n_runs=25
        )
        
        s = result["stability"]
        q = result["quality"]

        # Zbieranie metryk do podsumowania
        metrics = {
            "ARI": s["ARI_scores"],
            "NMI": s["NMI_scores"],
        }
        
        if q is not None:
            metrics["Silhouette"] = [q["silhouette"]]  # pojedynczy wynik, ale w liście
            metrics["Davies-Bouldin"] = [q["davies_bouldin"]]
            metrics["Calinski-Harabasz"] = [q["calinski_harabasz"]]

        all_results[name] = metrics
        
        # Podsumowanie dla tego algorytmu
        summary_df = summarize_metrics(metrics, algorithm_name=name)
        summary_dfs.append(summary_df)
        
        # Do porównań par
        ari_pairs.append((name, s["ARI_scores"]))
        nmi_pairs.append((name, s["NMI_scores"]))
        if q is not None:
            silhouette_pairs.append((name, [q["silhouette"]]))

    # Złączenie podsumowań
    summary_combined = pd.concat(summary_dfs, ignore_index=True)
    save_results(summary_combined, "stat_summary_metrics.csv")
    print("\n[✓] Zapisano podsumowanie metryk: stat_summary_metrics.csv")

    # Porównania par algorytmów
    ari_comp = compare_algorithms(ari_pairs, metric_name="ARI", alpha=0.05)
    nmi_comp = compare_algorithms(nmi_pairs, metric_name="NMI", alpha=0.05)
    silhouette_comp = compare_algorithms(silhouette_pairs, metric_name="Silhouette", alpha=0.05)

    comparisons_combined = pd.concat([ari_comp, nmi_comp, silhouette_comp], ignore_index=True)
    save_results(comparisons_combined, "stat_comparisons_pairwise.csv")
    print("[✓] Zapisano porównania par: stat_comparisons_pairwise.csv")

    # Porównania z testem Wilcoxona + poprawka Holma
    ari_comp_wilcoxon = compare_algorithms_wilcoxon_holm(ari_pairs, metric_name="ARI", alpha=0.05)
    nmi_comp_wilcoxon = compare_algorithms_wilcoxon_holm(nmi_pairs, metric_name="NMI", alpha=0.05)
    silhouette_comp_wilcoxon = compare_algorithms_wilcoxon_holm(silhouette_pairs, metric_name="Silhouette", alpha=0.05)

    comparisons_wilcoxon = pd.concat([ari_comp_wilcoxon, nmi_comp_wilcoxon, silhouette_comp_wilcoxon], ignore_index=True)
    save_results(comparisons_wilcoxon, "stat_comparisons_wilcoxon_holm.csv")
    print("[✓] Zapisano porównania Wilcoxona z poprawką Holma: stat_comparisons_wilcoxon_holm.csv")

    ranking = rank_algorithms(summary_combined)
    print("\n" + "="*50)
    print("RANKING ALGORYTMÓW")
    print("="*50)
    print(ranking.to_string(index=False))
    ranking.to_csv('stat_ranking_algorithms.csv', index=False)
    print("\nZapisano do: stat_ranking_algorithms.csv")

    print("\n=== PODSUMOWANIE ===")
    print(summary_combined.to_string(index=False))
    print("\n=== PORÓWNANIA PAR (Mann-Whitney U) ===")
    print(comparisons_combined.to_string(index=False))
    print("\n=== PORÓWNANIA PAR (Test Wilcoxona + Poprawka Holma) ===")
    print(comparisons_wilcoxon.to_string(index=False))


if __name__ == "__main__":
    main()
