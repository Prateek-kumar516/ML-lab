import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeGiniImpurity:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _gini(self, y):
        if len(y) == 0:
            return 0
        probs = np.bincount(y) / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain = 0
        best_split = None
        parent_gini = self._gini(y)

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                right = ~left

                if not np.any(left) or not np.any(right):
                    continue

                n = len(y)
                weighted_gini = (
                    np.sum(left) / n * self._gini(y[left]) +
                    np.sum(right) / n * self._gini(y[right])
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)

        return best_split

    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return Node(value=np.bincount(y).argmax())

        split = self._best_split(X, y)

        if split is None:
            return Node(value=np.bincount(y).argmax())

        feature, threshold = split
        left = X[:, feature] <= threshold
        right = ~left

        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(X[left], y[left], depth + 1),
            right=self._build_tree(X[right], y[right], depth + 1)
        )

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


class DecisionTreeEntropy:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _best_split(self, X, y):
        best_gain = 0
        best_split = None
        parent_entropy = self._entropy(y)

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                right = ~left

                if not np.any(left) or not np.any(right):
                    continue

                n = len(y)
                child_entropy = (
                    len(y[left]) / n * self._entropy(y[left]) +
                    len(y[right]) / n * self._entropy(y[right])
                )

                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)

        return best_split

    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return Node(value=np.unique(y, return_counts=True)[0][
                np.argmax(np.unique(y, return_counts=True)[1])
            ])

        split = self._best_split(X, y)

        if split is None:
            values, counts = np.unique(y, return_counts=True)
            return Node(value=values[np.argmax(counts)])

        feature, threshold = split
        left = X[:, feature] <= threshold
        right = ~left

        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(X[left], y[left], depth + 1),
            right=self._build_tree(X[right], y[right], depth + 1)
        )

    def fit(self, X, y):
        self.root = self._build_tree(np.asarray(X), np.asarray(y), 0)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([
            self._predict_one(x, self.root) for x in X
        ])