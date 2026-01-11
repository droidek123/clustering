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
    X, _, _ = load_city_dataset("polish_cities_2020.csv")

    algorithms = {
        "KMeans": (kmeans_clustering, {"n_clusters": 3}),
        "Hierarchical": (hierarchical_clustering, {"n_clusters": 3}),
        "DBSCAN": (dbscan_clustering, {"eps": 1.2, "min_samples": 2}),
        "Custom": (custom_clustering, {"n_clusters": 4}),
    }

    print("\nREALNE DANE – METRYKI + STABILNOŚĆ\n")

    names = []
    silhouettes = []
    ari_means = []

    for name, (fn, params) in algorithms.items():
        result = stability_with_metrics(
            clustering_fn=fn,
            X=X,
            fn_params=params,
            algorithm_name=name,
            n_runs=25
        )
        s = result["stability"]
        q = result["quality"]

        names.append(name)
        ari_means.append(s["ARI_mean"])
        silhouettes.append(q["silhouette"] if q is not None else 0.0)
        print(f"\n{name}")

        print(f"ARI: {s['ARI_mean']:.3f} ± {s['ARI_std']:.3f}")
        print(f"NMI: {s['NMI_mean']:.3f} ± {s['NMI_std']:.3f}")

        if q is not None:
            print(f"Silhouette: {q['silhouette']:.3f}")
            print(f"Davies-Bouldin: {q['davies_bouldin']:.3f}")
            print(f"Calinski-Harabasz: {q['calinski_harabasz']:.2f}")
        else:
            print("Brak sensownej jakości klasteryzacji.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, silhouettes)
    axes[0].set_title("Jakość klasteryzacji (Silhouette)")
    axes[0].grid(axis="y")

    axes[1].bar(names, ari_means)
    axes[1].set_title("Stabilność klasteryzacji (ARI)")
    axes[1].grid(axis="y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
