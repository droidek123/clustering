import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# KONFIGURACJA
# ======================================================
CSV_PATH = "benchmark_results.csv"
EPS_OPTIMAL = 2.55
N_RUNS = 25

# ======================================================
# WCZYTANIE DANYCH
# ======================================================
df = pd.read_csv(CSV_PATH)

# ======================================================
# 1. DBSCAN – STABILNOŚĆ vs EPS (OSOBNE WYKRESY)
# ======================================================
df_db = df[
    (df["algorithm"] == "DBSCAN") &
    (df["ARI_mean"].notna())
]

for min_samples in sorted(df_db["min_samples"].unique()):
    subset = (
        df_db[df_db["min_samples"] == min_samples]
        .sort_values("eps")
    )

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        subset["eps"],
        subset["ARI_mean"],
        yerr=subset["ARI_std"],
        marker="o",
        capsize=4
    )

    plt.axvline(
        EPS_OPTIMAL,
        linestyle="--",
        color="black",
        label=f"eps ≈ {EPS_OPTIMAL}"
    )

    plt.xlabel(
        "eps – promień sąsiedztwa\n"
        "(jednostki cech po standaryzacji)"
    )
    plt.ylabel(
        f"ARI (średnia ± std,\n{N_RUNS} powtórzeń)"
    )
    plt.title(
        f"DBSCAN – stabilność klasteryzacji\n"
        f"min_samples = {int(min_samples)}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======================================================
# 2. DBSCAN – ROZMIESZCZENIE KONFIGURACJI (SCATTER)
# ======================================================

plt.figure(figsize=(8, 6))

sc = plt.scatter(
    df_db["eps"],
    df_db["min_samples"],
    c=df_db["ARI_mean"],
    s=40 + 20 * df_db["n_clusters_found"],
    cmap="viridis",
    alpha=0.8
)

plt.colorbar(
    sc,
    label=f"ARI (średnia z {N_RUNS} powtórzeń)"
)

plt.axvline(
    EPS_OPTIMAL,
    linestyle="--",
    color="black",
    label=f"eps ≈ {EPS_OPTIMAL}"
)

plt.xlabel(
    "eps – promień sąsiedztwa\n"
    "(jednostki cech po standaryzacji)"
)
plt.ylabel("min_samples")
plt.title("DBSCAN – przestrzeń parametrów i jakość klasteryzacji")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================
# 4. ALGORYTMY CENTROIDOWE – OSOBNE WYKRESY
# ======================================================

df_centroid = df[
    (df["algorithm"].isin(["KMeans", "Hierarchical", "Custom"])) &
    (df["ARI_mean"].notna())
]

for algo in ["KMeans", "Hierarchical", "Custom"]:
    subset = (
        df_centroid[df_centroid["algorithm"] == algo]
        .sort_values("n_clusters")
    )

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        subset["n_clusters"],
        subset["ARI_mean"],
        yerr=subset["ARI_std"],
        marker="o",
        capsize=4
    )

    plt.xlabel("Liczba klastrów (n_clusters)")
    plt.ylabel(f"ARI (średnia ± std,\n{N_RUNS} powtórzeń)")
    plt.title(f"{algo} – wpływ liczby klastrów na stabilność")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======================================================
# 5. PORÓWNANIE ALGORYTMÓW – NAJLEPSZE KONFIGURACJE
# (bez trywialnego ARI = 1 dla DBSCAN)
# ======================================================
best = df[
    (df["ARI_mean"].notna()) &
    (
        (df["algorithm"] != "DBSCAN") |
        (
            (df["algorithm"] == "DBSCAN") &
            (df["ARI_mean"] < 0.999) &
            (df["n_clusters_found"] > 1)
        )
    )
].sort_values(
    "ARI_mean",
    ascending=False
).groupby(
    "algorithm"
).first()

plt.figure(figsize=(7, 5))
plt.bar(
    best.index,
    best["ARI_mean"],
    yerr=best["ARI_std"],
    capsize=6
)

plt.ylabel(
    f"ARI (średnia ± std,\n{N_RUNS} powtórzeń)"
)
plt.title("Porównanie algorytmów – najlepsze sensowne konfiguracje")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ======================================================
# 5. WPŁYW LICZBY CECH – JAKOŚĆ (SILHOUETTE)
# ======================================================
df_sil = df[df["silhouette"].notna()]

plt.figure(figsize=(8, 6))

for algo in ["KMeans", "Hierarchical", "Custom", "DBSCAN"]:
    subset = df_sil[df_sil["algorithm"] == algo]

    grouped = (
        subset.groupby("n_features")["silhouette"]
        .mean()
        .sort_index()
    )

    plt.plot(
        grouped.index,
        grouped.values,
        marker="o",
        label=algo
    )

plt.xlabel("Liczba cech w zbiorze danych")
plt.ylabel("Średnia wartość Silhouette")
plt.title("Wpływ liczby cech na jakość klasteryzacji")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()