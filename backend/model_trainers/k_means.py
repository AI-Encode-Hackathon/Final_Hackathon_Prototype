from sklearn.cluster import KMeans

def train_model(k, X_train):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_train)

    return model
