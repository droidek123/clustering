import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_city_dataset(csv_path):
    """
    Wczytuje dane miast z CSV i przygotowuje je do klasteryzacji.
    """
    df = pd.read_csv(csv_path)

    # kolumna z nazwą miasta
    city_col = df.columns[0]
    df = df.rename(columns={city_col: "city"})

    cities = df["city"].values
    X = df.drop(columns=["city"]).values
    feature_names = list(df.columns[1:])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, cities, feature_names
