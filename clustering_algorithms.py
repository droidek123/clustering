from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from custom_clustering import custom_clustering


def kmeans_clustering(X, n_clusters=4, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels

def hierarchical_clustering(X, n_clusters=4):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X)
    return labels

def dbscan_clustering(X, eps=0.3, min_samples=20):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def custom_centroid_clustering(X, n_clusters=4, random_state=42):
    labels = custom_clustering(X, n_clusters=n_clusters, random_state=random_state)
    return labels