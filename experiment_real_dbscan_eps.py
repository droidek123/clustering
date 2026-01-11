import numpy as np
import matplotlib.pyplot as plt

from city_data_loader import load_city_dataset
from clustering_algorithms import dbscan_clustering
from stability_analysis import clustering_stability


def main():
    X, _, _ = load_city_dataset("polish_cities_2020.csv")

    eps_values = np.linspace(0.5, 3.0, 12)
    ari_means = []
    ari_stds = []
    n_clusters_list = []

    print("\nDBSCAN – stabilność vs eps (dane rzeczywiste)\n")

    for eps in eps_values:
        labels = dbscan_clustering(X, eps=eps, min_samples=2)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters_list.append(n_clusters)

        if n_clusters < 2:
            ari_means.append(0.0)
            ari_stds.append(0.0)
            print(f"eps={eps:.2f} -> brak klastrów")
            continue

        result = clustering_stability(
            clustering_fn=dbscan_clustering,
            X=X,
            fn_params={"eps": eps, "min_samples": 2}
        )

        ari_means.append(result["ARI_mean"])
        ari_stds.append(result["ARI_std"])

        print(f"eps={eps:.2f} -> ARI={result['ARI_mean']:.3f}, klastry={n_clusters}")

    # Wykres
    plt.figure(figsize=(8, 5))
    plt.errorbar(eps_values, ari_means, yerr=ari_stds, marker="o", capsize=4)
    plt.xlabel("eps")
    plt.ylabel("ARI (mean ± std)")
    plt.title("DBSCAN – stabilność vs eps (miasta 2020)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
