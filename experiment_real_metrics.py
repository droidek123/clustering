import matplotlib.pyplot as plt

from city_data_loader import load_city_dataset
from clustering_algorithms import (dbscan_clustering, hierarchical_clustering,
                                   kmeans_clustering)
from clustering_metrics import evaluate_clustering, print_metrics
from custom_clustering import custom_clustering


def main():
    X, _, _ = load_city_dataset("data/polish_cities_2020.csv")

    algorithms = {
        "KMeans": (kmeans_clustering, {"n_clusters": 4}),
        "Hierarchical": (hierarchical_clustering, {"n_clusters": 3}),
        "DBSCAN": (dbscan_clustering, {"eps": 2.32, "min_samples": 2}),
        "Custom (Mean Shift)": (custom_clustering, {}),
    }

    algo_names = []
    silhouettes = []

    for name, (fn, params) in algorithms.items():
        labels = fn(X, **params)
        metrics = evaluate_clustering(X, labels, name)
        print_metrics(metrics)

        if metrics is not None:
            algo_names.append(name)
            silhouettes.append(metrics["silhouette"])

    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, silhouettes)
    plt.ylabel("Silhouette score")
    plt.title("Porównanie jakości klasteryzacji – miasta 2020")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
