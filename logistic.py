import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data (same as before)
def load_and_preprocess_data(file_path, target_column, add_bias=True):
    df = pd.read_csv(file_path)

      # Special case: robot dataset with time series data
    if "robot" in file_path.lower():
        df = extract_robot_features(df, target_column)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Move target column to the end
    target = df.pop(target_column)
    df[target_column] = target

    # Label encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col], _ = pd.factorize(df[col])
            # Convert target to binary (0 and 1)
    if df[target_column].nunique() > 2:
        df[target_column] = (df[target_column] != 0).astype(float)

    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(float)

    # Min-Max Normalization
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # Add bias term
    if add_bias:
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Shuffle and split (80-20)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split_idx = int(0.8 * len(X))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# Extract features from robot dataset
def extract_robot_features(df, target_column):
    signal_types = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    features = {}

    for sig in signal_types:
        sig_cols = [col for col in df.columns if col.startswith(sig)]
        values = df[sig_cols].values
        features[f'{sig}_mean'] = values.mean(axis=1)
        features[f'{sig}_var'] = values.var(axis=1)

    features_df = pd.DataFrame(features)
    features_df[target_column] = df[target_column].values
    return features_df

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression training using Gradient Descent
def logistic_regression_train(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for epoch in range(epochs):
        linear_model = X @ weights
        y_pred = sigmoid(linear_model)

        # Binary cross-entropy loss
        error = y_pred - y
        gradient = (1 / n_samples) * (X.T @ error)
        weights -= lr * gradient

        if epoch % 100 == 0:
    # Clip predictions to avoid log(0)
            y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return weights

# Prediction
def logistic_regression_predict(X, weights):
    probs = sigmoid(X @ weights)
    return (probs >= 0.5).astype(int)

# Evaluation
def evaluate_classification(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    return f"Accuracy: {accuracy:.4f}"


if __name__ == "__main__":

    #for Glass dataset
    file_path = "datasets/glass_details_dataset.csv"
    target_col = "Class"

    #for Heart Failure dataset
    # file_path = "datasets/heart_failure_clinical_records_dataset.csv"
    # target_col = "DEATH_EVENT"

    # #for Robot Execution Failures dataset
    # file_path = "datasets/robot_dataset.csv"
    # target_col = "Class"

    # #for Wisconsin Diagnostic Breast Cancer dataset
    # file_path = "datasets/wisconsin_diagnostic_breast_cancer_dataset.csv"
    # target_col = "Diagnosis"

    # #for Mushroom dataset
    # file_path = "datasets/mushroom_dataset.csv"
    # target_col = "class"

    # Step 1: Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_col)

    # Step 2: Train
    weights = logistic_regression_train(X_train, y_train, lr=0.01, epochs=1000)

    # Step 3: Predict
    y_pred = logistic_regression_predict(X_test, weights)

    # Step 4: Evaluate
    result = evaluate_classification(y_test, y_pred)
    print(result)
   
