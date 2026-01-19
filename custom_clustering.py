import numpy as np
from scipy.spatial.distance import euclidean


class MeanShiftClustering:
    """
    Mean Shift Clustering Algorithm
    
    Źródło: Comaniciu, D., & Meer, P. (2002). 
    "Mean Shift: A Robust Approach toward Feature Space Analysis"
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(5), 603-619.
    
    Algorytm kernel-based, który nie wymaga z góry znania liczby klastrów.
    Iteracyjnie przesuwa każdy punkt w kierunku modu (lokalnego maksimum)
    rozkładu gęstości.
    """
    
    def __init__(self, bandwidth=1.0, max_iter=100, tol=1e-4, random_state=None):
        """
        Parametry:
        -----------
        bandwidth : float
            Parametr szerokości kernela (sigma dla Gaussiana)
        max_iter : int
            Maksymalna liczba iteracji dla konwergencji każdego punktu
        tol : float
            Tolerancja dla warunku konwergencji
        random_state : int
            Seed dla reprodukowalności
        """
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        
    def _gaussian_kernel(self, distance):
        """Gaussowski kernel dla ważenia punktów"""
        return np.exp(-(distance ** 2) / (2 * self.bandwidth ** 2))
    
    def _mean_shift_single_point(self, point, X):
        """
        Wykonuje Mean Shift dla pojedynczego punktu
        Iteracyjnie przesuwa punkt w kierunku średniej ważonej sąsiadów
        """
        for _ in range(self.max_iter):
            # Oblicz odległości od wszystkich punktów
            distances = np.linalg.norm(X - point, axis=1)
            
            # Zastosuj kernel (wagi)
            weights = self._gaussian_kernel(distances)
            
            # Oblicz nową pozycję jako średnią ważoną
            new_point = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            # Sprawdź konwergencję
            shift = np.linalg.norm(new_point - point)
            point = new_point
            
            if shift < self.tol:
                break
        
        return point
    
    def _find_cluster_labels(self, X, cluster_centers, radius=None):
        """Przypisz punkty do najbliższych modów (cluster centers)"""
        if radius is None:
            radius = self.bandwidth * 2
        
        labels = np.zeros(X.shape[0], dtype=int)
        
        for i, point in enumerate(X):
            distances = np.linalg.norm(cluster_centers - point, axis=1)
            labels[i] = np.argmin(distances)
        
        return labels
    
    def fit(self, X):
        """Dopasuj model Mean Shift do danych"""
        np.random.seed(self.random_state)
        
        # Dla każdego punktu, oblicz do którego modu dochodzi
        shifted_points = []
        
        for point in X:
            shifted_point = self._mean_shift_single_point(point.copy(), X)
            shifted_points.append(shifted_point)
        
        shifted_points = np.array(shifted_points)
        
        # Grupuj punkty które zbiegły do tego samego modu
        # Łącz punkty które są bliskie siebie
        cluster_centers = []
        used = np.zeros(len(shifted_points), dtype=bool)
        
        for i, point in enumerate(shifted_points):
            if used[i]:
                continue
            
            # Znaleźć wszystkie punkty blisko tego modu
            distances = np.linalg.norm(shifted_points - point, axis=1)
            nearby = distances < self.bandwidth
            used[nearby] = True
            
            # Średnia tych punktów jako centrum
            cluster_center = shifted_points[nearby].mean(axis=0)
            cluster_centers.append(cluster_center)
        
        self.cluster_centers_ = np.array(cluster_centers)
        self.labels_ = self._find_cluster_labels(X, self.cluster_centers_)
        
        return self
    
    def predict(self, X):
        """Przewiduj etykiety dla nowych punktów"""
        if self.cluster_centers_ is None:
            raise ValueError("Model nie został dopasowany. Uruchom fit() najpierw.")
        
        return self._find_cluster_labels(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        """Dopasuj model i zwróć etykiety"""
        self.fit(X)
        return self.labels_


def custom_clustering(X, bandwidth=None, random_state=42):
    """
    Mean Shift Clustering - niestandardowa metoda
    
    Parametry:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dane do klastrowania
    bandwidth : float, optional
        Parametr szerokości kernela. Jeśli None, obliczony z regułą Silvermana
    random_state : int
        Seed dla reprodukowalności
    
    Zwraca:
    --------
    labels : array, shape (n_samples,)
        Etykiety klastrów dla każdego punktu
    """
    # Jeśli bandwidth nie podany, użyj reguły Silvermana z skalowaniem
    if bandwidth is None:
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # Reguła Silvermana
        bandwidth = (n_samples * (n_features + 2) / 4.) ** (-1. / (n_features + 4))
        # Skaluj na wariancję danych
        bandwidth *= np.std(X)
        # Zwiększ bandwidth aby uniknąć nadmiernej fragmentacji
        bandwidth *= 1.5
    
    clustering = MeanShiftClustering(
        bandwidth=bandwidth,
        max_iter=100,
        random_state=random_state
    )
    labels = clustering.fit_predict(X)
    return labels