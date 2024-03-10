from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def train_model(k, X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model
