from sklearn.tree import DecisionTreeClassifier
import numpy as np

def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model
