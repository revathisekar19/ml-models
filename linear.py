import pandas as pd
import numpy as np

# Load and preprocess the data
def load_and_preprocess_data(file_path, target_column, add_bias=True):
    # Load dataset
    df = pd.read_csv(file_path)

    # Move target column to the end
    target = df.pop(target_column)
    df[target_column] = target

    # Label encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col], _ = pd.factorize(df[col])

    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(float)

    # Min-max normalization
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # Add bias column
    if add_bias:
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))

    # Shuffle and split (80-20)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split_idx = int(0.8 * len(X))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# Linear Regression using Gradient Descent
def linear_regression_train_gd(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for epoch in range(epochs):
        predictions = X @ weights
        errors = predictions - y
        gradients = (2 / n_samples) * X.T @ errors
        weights -= lr * gradients

        if epoch % 100 == 0:
            mse = np.mean(errors ** 2)
            print(f"Epoch {epoch}: MSE = {mse:.4f}")

    return weights

def linear_regression_predict(X, weights):
    return X @ weights

def evaluate_regression(y_true, y_pred, is_classification=False):
    if is_classification:
        y_pred_class = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y_pred_class == y_true)
        return f"Accuracy: {accuracy:.4f}"
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        return f"MSE: {mse:.4f}"
    
def confusion_matrix(y_true, y_pred_class):
    unique_classes = np.unique(np.concatenate((y_true, y_pred_class)))
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    class_to_index = {label: idx for idx, label in enumerate(unique_classes)}

    for true, pred in zip(y_true, y_pred_class):
        i = class_to_index[true]
        j = class_to_index[pred]
        matrix[i, j] += 1

    return matrix, unique_classes


# Main
if __name__ == "__main__":
    file_path = "datasets/glass_details_dataset.csv"
    target_col = "Class"

    # Step 1: Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_col)

    # Step 2: Train
    weights = linear_regression_train_gd(X_train, y_train, lr=0.01, epochs=1000)

    # Step 3: Predict
    y_pred = linear_regression_predict(X_test, weights)

    # Step 4: Evaluate
    result = evaluate_regression(y_test, y_pred, is_classification=True)
    print("Result of the linear model:", result)


