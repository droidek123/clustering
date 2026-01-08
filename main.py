import matplotlib.pyplot as plt

from clustering_algorithms import (custom_centroid_clustering,
                                   dbscan_clustering, hierarchical_clustering,
                                   kmeans_clustering)
from clustering_metrics import evaluate_clustering, print_metrics
from data_generator import generate_data

# Generowanie danych
X_scaled, y_true = generate_data(n_samples=500, centers=4, n_features=2, cluster_std=0.85, random_state=0)

# Wykonanie algorytmów
y_kmeans = kmeans_clustering(X_scaled, n_clusters=4)
y_hierarchical = hierarchical_clustering(X_scaled, n_clusters=4)
y_dbscan = dbscan_clustering(X_scaled, eps=0.3, min_samples=20)
y_custom = custom_centroid_clustering(X_scaled, n_clusters=4)

# Ocena wyników
metrics_kmeans = evaluate_clustering(X_scaled, y_kmeans, "K-Means")
metrics_hierarchical = evaluate_clustering(X_scaled, y_hierarchical, "Hierarchical Clustering")
metrics_dbscan = evaluate_clustering(X_scaled, y_dbscan, "DBSCAN")
metrics_custom = evaluate_clustering(X_scaled, y_custom, "Custom Centroid Clustering")

# Wyświetlenie metryk
print_metrics(metrics_kmeans)
print_metrics(metrics_hierarchical)
print_metrics(metrics_dbscan)
print_metrics(metrics_custom)

# Wizualizacja
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# K-Means
axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50, edgecolor='k')
axes[0, 0].set_title(f"K-Means (K=4)\nSilhouette: {metrics_kmeans['silhouette']:.3f}")
axes[0, 0].set_xlabel("Cecha 1")
axes[0, 0].set_ylabel("Cecha 2")

# Hierarchical
axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_hierarchical, cmap='viridis', s=50, edgecolor='k')
axes[0, 1].set_title(f"Hierarchical Clustering (K=4)\nSilhouette: {metrics_hierarchical['silhouette']:.3f}")
axes[0, 1].set_xlabel("Cecha 1")
axes[0, 1].set_ylabel("Cecha 2")

# DBSCAN
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
silhouette_dbscan = metrics_dbscan['silhouette'] if metrics_dbscan else 0
axes[1, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='viridis', s=50, edgecolor='k')
axes[1, 0].set_title(f"DBSCAN (eps=0.3, min=20)\nKlastry: {n_clusters}, Silhouette: {silhouette_dbscan:.3f}")
axes[1, 0].set_xlabel("Cecha 1")
axes[1, 0].set_ylabel("Cecha 2")

# Custom Centroid
axes[1, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_custom, cmap='viridis', s=50, edgecolor='k')
axes[1, 1].set_title(f"Custom Centroid Clustering (K=4)\nSilhouette: {metrics_custom['silhouette']:.3f}")
axes[1, 1].set_xlabel("Cecha 1")
axes[1, 1].set_ylabel("Cecha 2")

plt.show()