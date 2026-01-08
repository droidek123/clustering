import numpy as np


class SimpleCentroidClustering:
    """
    Prosty algorytm klasteryzacji oparty na środkach ciężkości.
    Podobny do K-Means, ale z inną metodą aktualizacji centroidów.
    """
    
    def __init__(self, n_clusters=4, max_iter=100, tol=1e-4, random_state=None):
        """
        Args:
            n_clusters: liczba klastrów
            max_iter: maksymalna liczba iteracji
            tol: tolerancja dla zbieżności
            random_state: ziarno losowości
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        
    def _initialize_centroids(self, X):
        """Inicjalizacja centroidów - losowe punkty z danych."""
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def _assign_clusters(self, X):
        """Przypisanie punktów do najbliższych centroidów."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Aktualizacja centroidów jako średnia punktów w klastrze."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Jeśli klaster pusty, zachowaj stary centroid
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def fit(self, X):
        """Trenowanie modelu."""
        # Inicjalizacja centroidów
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Przypisanie klastrów
            labels = self._assign_clusters(X)
            
            # Aktualizacja centroidów
            new_centroids = self._update_centroids(X, labels)
            
            # Sprawdzenie zbieżności
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        self.labels_ = labels
        return self
    
    def predict(self, X):
        """Predykcja klastrów dla nowych danych."""
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Trenowanie i predykcja."""
        self.fit(X)
        return self.labels_


def custom_clustering(X, n_clusters=4, random_state=42):
    """Wrapper dla własnego algorytmu klasteryzacji."""
    clustering = SimpleCentroidClustering(
        n_clusters=n_clusters,
        max_iter=100,
        random_state=random_state
    )
    labels = clustering.fit_predict(X)
    return labels