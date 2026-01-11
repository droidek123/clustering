import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from clustering_algorithms import dbscan_clustering
from stability_analysis import clustering_stability


X, _ = make_blobs(
    n_samples=600,
    centers=4,
    cluster_std=1.3,
    random_state=42
)

eps_values = np.linspace(0.3, 2.0, 10)

ari_means = []
ari_stds = []

for eps in eps_values:
    result = clustering_stability(
        clustering_fn=dbscan_clustering,
        X=X,
        fn_params={"eps": eps, "min_samples": 6}
    )

    ari_means.append(result["ARI_mean"])
    ari_stds.append(result["ARI_std"])


plt.figure(figsize=(8, 5))
plt.errorbar(
    eps_values,
    ari_means,
    yerr=ari_stds,
    marker="o",
    capsize=4
)
plt.xlabel("eps")
plt.ylabel("ARI")
plt.title("Stabilność DBSCAN w funkcji parametru eps")
plt.grid(True)
plt.tight_layout()
plt.show()
