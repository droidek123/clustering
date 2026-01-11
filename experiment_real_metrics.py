from city_data_loader import load_city_dataset
from clustering_metrics import evaluate_clustering, print_metrics
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

    # LISTY DO WYKRESU
    algo_names = []
    silhouettes = []

    for name, (fn, params) in algorithms.items():
        labels = fn(X, **params)
        metrics = evaluate_clustering(X, labels, name)
        print_metrics(metrics)

        if metrics is not None:
            algo_names.append(name)
            silhouettes.append(metrics["silhouette"])

    # WYKRES
    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, silhouettes)
    plt.ylabel("Silhouette score")
    plt.title("Porównanie jakości klasteryzacji – miasta 2020")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
