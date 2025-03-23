import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = None
        self.feature_types = []  # Tracks if features are categorical

    def _is_categorical(self, X):
        return [True if len(np.unique(X[:, i])) < 10 else False for i in range(X.shape[1])]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.feature_types = self._is_categorical(X)

        # Compute class priors
        for c in self.classes:
            self.class_priors[c] = np.mean(y == c)

        self.feature_likelihoods = defaultdict(dict)

        for idx in range(n_features):
            if self.feature_types[idx]:  # Categorical
                for c in self.classes:
                    X_c = X[y == c, idx]
                    values, counts = np.unique(X_c, return_counts=True)
                    probs = counts / counts.sum()
                    self.feature_likelihoods[(idx, c)] = dict(zip(values, probs))
            else:  # Continuous (Gaussian)
                for c in self.classes:
                    X_c = X[y == c, idx]
                    mean = np.mean(X_c)
                    std = np.std(X_c) + 1e-6
                    self.feature_likelihoods[(idx, c)] = (mean, std)

    def _gaussian_pdf(self, x, mean, std):
        exponent = np.exp(- ((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prob = np.log(self.class_priors[c])
                for idx in range(len(x)):
                    val = x[idx]
                    if self.feature_types[idx]:  # Categorical
                        likelihoods = self.feature_likelihoods.get((idx, c), {})
                        prob += np.log(likelihoods.get(val, 1e-6))
                    else:  # Continuous
                        mean, std = self.feature_likelihoods[(idx, c)]
                        prob += np.log(self._gaussian_pdf(val, mean, std))
                class_probs[c] = prob
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

# ----------------------
# Helper functions
# ----------------------

def evaluate_model(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return f"Accuracy: {accuracy:.4f}"

def load_and_run_naive_bayes(file_path, target_column):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    y = df.pop(target_column)
    df[target_column] = y

    # Encode categorical string features
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col], _ = pd.factorize(df[col])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nNaive Bayes Classifier Results:")
    print(evaluate_model(y_test, y_pred))

# ----------------------
# Run the script
# ----------------------

if __name__ == "__main__":

    #for Heart Failure dataset
    load_and_run_naive_bayes("datasets/heart_failure_clinical_records_dataset.csv", "DEATH_EVENT")

    #for Robot Execution Failures dataset
    load_and_run_naive_bayes("datasets/robot_execution_failures_dataset.csv","Class")

    #for Wisconsin Diagnostic Breast Cancer dataset
    load_and_run_naive_bayes("datasets/wisconsin_diagnostic_breast_cancer_dataset.csv","Diagnosis")

    #for Mushroom dataset
    load_and_run_naive_bayes("datasets/mushroom_dataset.csv","class")

    #for Glass dataset
    load_and_run_naive_bayes("datasets/glass_details_dataset.csv","Class")
    
