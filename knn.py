import pandas as pd
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, task="classification"):
        self.k = k
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_point(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        if self.task == "classification":
            most_common = Counter(k_labels).most_common(1)[0][0]
            return most_common
        else:
            return np.mean(k_labels)

    def predict(self, X):
        return np.array([self._predict_point(x) for x in X])

def is_classification(y):
    return pd.api.types.is_integer_dtype(y) or pd.api.types.is_object_dtype(y)

def evaluate_model(y_true, y_pred, task):
    if task == "classification":
        accuracy = np.mean(y_true == y_pred)
        return f"Accuracy: {accuracy:.4f}"
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        return f"MSE: {mse:.4f}"

def load_and_run_knn(file_path, target_column, k=3):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    y = df.pop(target_column)
    df[target_column] = y

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col], _ = pd.factorize(df[col])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    task = "classification" if is_classification(y) else "regression"

    # Min-Max Normalization
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = KNN(k=k, task=task)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nKNN ({task.title()}) Results with k={k}:")
    print(evaluate_model(y_test, y_pred, task))

# 🔍 Example call:
if __name__ == "__main__":
    load_and_run_knn("datasets/heart_failure_clinical_records_dataset.csv", "DEATH_EVENT", k=5)
