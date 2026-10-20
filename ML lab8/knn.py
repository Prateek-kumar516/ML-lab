import numpy as np

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    def predict(self, X):
        predictions = []

        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)

            idx = np.argpartition(
                distances, self.n_neighbors - 1
            )[:self.n_neighbors]

            labels = self.y_train[idx]
            prediction = np.bincount(labels).argmax()
            predictions.append(prediction)

        return np.array(predictions)