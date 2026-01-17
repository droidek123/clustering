from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)


def evaluate_clustering(X, labels, algorithm_name):
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    n_clusters = len(set(labels_filtered))
    
    if n_clusters < 2:
        return None
    
    silhouette = silhouette_score(X_filtered, labels_filtered)
    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
    calinski_harabasz = calinski_harabasz_score(X_filtered, labels_filtered)
    
    metrics = {
        'algorithm': algorithm_name,
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }
    
    return metrics

def print_metrics(metrics):
    if metrics is None:
        return
    
    print(f"\n{'='*50}")
    print(f"Algorytm: {metrics['algorithm']}")
    print(f"Liczba klastrów: {metrics['n_clusters']}")
    print(f"{'='*50}")
    print(f"Silhouette Score:        {metrics['silhouette']:.4f} (wyższy = lepiej, zakres [-1, 1])")
    print(f"Davies-Bouldin Index:    {metrics['davies_bouldin']:.4f} (niższy = lepiej)")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f} (wyższy = lepiej)")