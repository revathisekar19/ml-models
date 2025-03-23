import pandas as pd
import numpy as np


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, task, max_depth=10, min_samples_split=2):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or len(X) < self.min_samples_split or depth >= self.max_depth:
            return DecisionTreeNode(value=self._leaf_value(y))

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return DecisionTreeNode(value=self._leaf_value(y))

        left_idx = X[:, best_feat] < best_thresh
        right_idx = ~left_idx

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return DecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_score = -float("inf") if self.task == "classification" else float("inf")
        best_feat, best_thresh = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for thresh in thresholds:
                left_idx = X[:, feature_index] < thresh
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                if self.task == "classification":
                    score = self._information_gain(y, y[left_idx], y[right_idx])
                    if score > best_score:
                        best_score = score
                        best_feat, best_thresh = feature_index, thresh
                else:
                    score = self._mse(y[left_idx], y[right_idx])
                    if score < best_score:
                        best_score = score
                        best_feat, best_thresh = feature_index, thresh

        return best_feat, best_thresh

    def _information_gain(self, parent, left, right):
        def entropy(y):
            classes, counts = np.unique(y, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-9))

        total = len(parent)
        return entropy(parent) - (
            len(left) / total * entropy(left) + len(right) / total * entropy(right)
        )

    def _mse(self, left, right):
        def mse(group):
            return np.mean((group - np.mean(group)) ** 2)
        total = len(left) + len(right)
        return (len(left) / total * mse(left)) + (len(right) / total * mse(right))

    def _leaf_value(self, y):
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return np.mean(y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

# task type detection
def determine_task_type(y):
    unique_vals = np.unique(y)
    if y.dtype == 'object' or len(unique_vals) <= 10:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and len(unique_vals) <= 10:
        return "classification"
    return "regression"

def evaluate_model(y_true, y_pred, task):
    if task == "classification":
        acc = np.mean(y_true == y_pred)
        return f"Accuracy: {acc:.4f}"
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        return f"MSE: {mse:.4f}"


def load_and_run_decision_tree(file_path, target_column):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    y = df.pop(target_column)
    df[target_column] = y

    # Encode categorical features
    for col in df.columns:
        # if df[col].dtype == 'object':
        #     df[col], _ = pd.factorize(df[col])
        if df[col].dtype == 'object' or col == target_column:
            df[col], _ = pd.factorize(df[col].astype(str))


    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    task = determine_task_type(y)
    print(f"Detected task type: {task}")

    # Shuffle and split
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = DecisionTree(task=task, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nDecision Tree ({task.title()}) Results:")
    print(evaluate_model(y_test, y_pred, task))


if __name__ == "__main__":

    #for Heart Failure Dataset
    load_and_run_decision_tree("datasets/heart_failure_clinical_records_dataset.csv", "DEATH_EVENT")

    #for Robot Execution Failures Dataset
    load_and_run_decision_tree("datasets/robot_execution_failures_dataset.csv","Class")

    #for Wisconsin Diagnostic Breast Cancer Dataset
    load_and_run_decision_tree("datasets/wisconsin_diagnostic_breast_cancer_dataset.csv","Diagnosis")

    #for Mushroom Dataset
    load_and_run_decision_tree("datasets/mushroom_dataset.csv","class")

    #for Glass Dataset
    load_and_run_decision_tree("datasets/glass_details_dataset.csv","Class")
