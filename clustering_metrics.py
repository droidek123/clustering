from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)


def evaluate_clustering(X, labels, algorithm_name):
    """
    Ocena jakości klasteryzacji za pomocą trzech metryk.
    
    Args:
        X: dane wejściowe
        labels: etykiety klastrów
        algorithm_name: nazwa algorytmu
    
    Returns:
        dict: słownik z metrykami
    """
    # Filtrujemy punkty szumu dla DBSCAN (etykiety -1)
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    # Sprawdzamy czy mamy wystarczająco klastrów
    n_clusters = len(set(labels_filtered))
    
    if n_clusters < 2:
        print(f"\n{algorithm_name}: Za mało klastrów do oceny (znaleziono: {n_clusters})")
        return None
    
    # Obliczanie metryk
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
    """Wyświetla metryki w czytelny sposób."""
    if metrics is None:
        return
    
    print(f"\n{'='*50}")
    print(f"Algorytm: {metrics['algorithm']}")
    print(f"Liczba klastrów: {metrics['n_clusters']}")
    print(f"{'='*50}")
    print(f"Silhouette Score:        {metrics['silhouette']:.4f} (wyższy = lepiej, zakres [-1, 1])")
    print(f"Davies-Bouldin Index:    {metrics['davies_bouldin']:.4f} (niższy = lepiej)")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f} (wyższy = lepiej)")