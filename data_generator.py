from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples=500, centers=4, n_features=2, cluster_std=0.85, random_state=0):
    """
    Generuje dane dla algorytmów klasteryzacji.
    
    Args:
        n_samples: liczba próbek
        centers: liczba naturalnych skupisk
        n_features: liczba cech (wymiarów)
        cluster_std: odchylenie standardowe klastrów
        random_state: ziarno losowości
    
    Returns:
        X_scaled: przeskalowane dane
        y_true: prawdziwe etykiety (do porównania)
    """
    X, y_true = make_blobs(n_samples=n_samples, 
                          centers=centers,
                          n_features=n_features,
                          cluster_std=cluster_std, 
                          random_state=random_state)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true