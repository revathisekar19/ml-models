
import pandas as pd
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        concat_size = input_size + hidden_size

        # Weight matrices
        self.Wf = np.random.randn(hidden_size, concat_size) * 0.1
        self.Wi = np.random.randn(hidden_size, concat_size) * 0.1
        self.Wo = np.random.randn(hidden_size, concat_size) * 0.1
        self.Wc = np.random.randn(hidden_size, concat_size) * 0.1

        # Bias vectors
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        # Output layer
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def forward(self, X):
        _, timesteps, _ = X.shape
        h_t = np.zeros((self.hidden_size, 1))
        C_t = np.zeros((self.hidden_size, 1))

        for t in range(timesteps):
            x_t = X[:, t, :].reshape(-1, 1)
            x_combined = np.vstack((h_t, x_t))

            f_t = self.sigmoid(self.Wf @ x_combined + self.bf)
            i_t = self.sigmoid(self.Wi @ x_combined + self.bi)
            o_t = self.sigmoid(self.Wo @ x_combined + self.bo)
            C_tilde = self.tanh(self.Wc @ x_combined + self.bc)

            C_t = f_t * C_t + i_t * C_tilde
            h_t = o_t * self.tanh(C_t)

        y = self.Wy @ h_t + self.by
        return self.softmax(y).flatten()



def load_robot_lstm_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Extract labels
    y = df["Class"]
    if y.dtype == 'object':
        y, _ = pd.factorize(y)

    # Create time series 
    features = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    n_timesteps = 15
    X_seq = []

    for i in range(len(df)):
        sample = []
        for feat in features:
            values = [df.iloc[i][f"{feat}{t+1}"] for t in range(n_timesteps)]
            sample.append(values)
        sample = np.array(sample).T  
        X_seq.append(sample)

    X_seq = np.array(X_seq)  
    y = np.array(y)

    return X_seq, y


if __name__ == "__main__":
    # Load data
    X, y = load_robot_lstm_data("datasets/robot_dataset.csv")

    n_classes = len(np.unique(y))
model = LSTM(input_size=6, hidden_size=32, output_size=n_classes)

# Inference
predictions = []
for i in range(len(X)):
    probs = model.forward(X[i:i+1])
    pred_class = np.argmax(probs)
    predictions.append(pred_class)

predictions = np.array(predictions)
accuracy = np.mean(predictions == y)
print(f"\nLSTM (Multiclass, Forward-only) Accuracy: {accuracy:.4f}")

