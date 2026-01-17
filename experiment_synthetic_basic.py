import matplotlib.pyplot as plt

from clustering_algorithms import (dbscan_clustering, hierarchical_clustering,
                                   kmeans_clustering)
from clustering_metrics import evaluate_clustering, print_metrics
from custom_clustering import custom_clustering
from data_generator import generate_data


def main():
    X, _ = generate_data(
        n_samples=500,
        centers=4,
        n_features=2,
        cluster_std=0.85,
        random_state=0
    )

    algorithms = {
        "KMeans": (kmeans_clustering, {"n_clusters": 4}),
        "Hierarchical": (hierarchical_clustering, {"n_clusters": 4}),
        "DBSCAN": (dbscan_clustering, {"eps": 0.3, "min_samples": 20}),
        "Custom": (custom_clustering, {"n_clusters": 4}),
    }

    results = {}

    for name, (fn, params) in algorithms.items():
        labels = fn(X, **params)
        metrics = evaluate_clustering(X, labels, name)
        print_metrics(metrics)
        results[name] = labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (name, labels) in zip(axes, results.items()):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40)
        ax.set_title(name)
        ax.set_xlabel("Cecha 1")
        ax.set_ylabel("Cecha 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
