from sklearn.neural_network import MLPClassifier

def train_model(X_train, y_train):
    model = MLPClassifier(
        max_iter=2000,
        batch_size=32,
        random_state=42,
        activation='logistic',
        alpha=5,
        epsilon=1e-08,
        hidden_layer_sizes=(100, ),
        learning_rate="adaptive",
        solver='adam'
    )

    # train the model
    model.fit(X_train, y_train)
    return model