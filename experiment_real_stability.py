from city_data_loader import load_city_dataset
from stability_analysis import stability_with_metrics
import matplotlib.pyplot as plt

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
        "Custom (Mean Shift)": (custom_clustering, {}),
    }

    names = []
    ari_means = []
    ari_stds = []

    for name, (fn, params) in algorithms.items():
        result = stability_with_metrics(
            clustering_fn=fn,
            X=X,
            fn_params=params,
            algorithm_name=name,
            n_runs=25
        )
        s = result["stability"]

        names.append(name)
        ari_means.append(s["ARI_mean"])
        ari_stds.append(s["ARI_std"])

        print(f"\n{name}")
        stability = result["stability"]
        quality = result["quality"]

        print(f"ARI: {stability['ARI_mean']:.3f} ± {stability['ARI_std']:.3f}")
        print(f"NMI: {stability['NMI_mean']:.3f} ± {stability['NMI_std']:.3f}")

        if quality is not None:
            print(f"Silhouette: {quality['silhouette']:.3f}")
            print(f"Davies-Bouldin: {quality['davies_bouldin']:.3f}")
            print(f"Calinski-Harabasz: {quality['calinski_harabasz']:.2f}")
        else:
            print("Brak sensownej jakości klasteryzacji (np. brak klastrów).")

    plt.figure(figsize=(8, 5))
    plt.errorbar(names, ari_means, yerr=ari_stds, fmt="o", capsize=5)
    plt.ylabel("ARI (mean ± std)")
    plt.title("Stabilność klasteryzacji – miasta 2020")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
