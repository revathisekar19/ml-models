import matplotlib.pyplot as plt

# Accuracy values per model from the extracted PDF content
models = ["Linear Regression", "Logistic Regression", "KNN", "Naive Bayes", "Decision Tree"]

# Accuracy for each dataset per model
datasets = ["Breast Cancer", "Mushroom", "Robot Failures", "Heart Failure", "Glass"]

accuracy_data = {
    "Linear Regression": [0.9298, 0.9237, 0.6250, 0.8000, 0.3953],
    "Logistic Regression": [0.8509, 0.8702, 0.8750, 0.6833, 1.0000],
    "KNN": [0.9825, 1.0000, 0.7500, 0.7333, 0.7209],
    "Naive Bayes": [0.9298, 0.9852, 0.9583, 0.8167, 0.6047],
    "Decision Tree": [0.9649, 0.9988, 0.8833, 0.7333, 0.7442]
}

# Plot a bar chart for each model
for model in models:
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, accuracy_data[model])
    plt.ylim(0, 1.1)
    plt.title(f"{model} Accuracy on Different Datasets")
    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



