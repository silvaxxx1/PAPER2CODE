import numpy as np

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# MLP Layer
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        # Forward pass
        self.h = np.dot(x, self.W1) + self.b1
        self.h_relu = np.maximum(0, self.h)  # ReLU activation
        self.out = np.dot(self.h_relu, self.W2) + self.b2
        return self.out

    def predict(self, x):
        logits = self.forward(x)
        return softmax(logits)

# LSTM Layer
class LSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = np.random.randn(input_size, hidden_size) * 0.01
        self.W_f = np.random.randn(input_size, hidden_size) * 0.01
        self.W_o = np.random.randn(input_size, hidden_size) * 0.01
        self.W_c = np.random.randn(input_size, hidden_size) * 0.01
        self.U_i = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_f = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_o = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_c = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((1, hidden_size))
        self.b_f = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))
        self.b_c = np.zeros((1, hidden_size))

    def forward(self, x, h_prev, c_prev):
        # Forward pass
        self.i = sigmoid(np.dot(x, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i)
        self.f = sigmoid(np.dot(x, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f)
        self.o = sigmoid(np.dot(x, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o)
        self.c_hat = tanh(np.dot(x, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c)
        self.c = self.f * c_prev + self.i * self.c_hat
        self.h = self.o * tanh(self.c)
        return self.h, self.c

# GRU Layer
class GRU:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = np.random.randn(input_size, hidden_size) * 0.01
        self.W_r = np.random.randn(input_size, hidden_size) * 0.01
        self.W_h = np.random.randn(input_size, hidden_size) * 0.01
        self.U_z = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_r = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((1, hidden_size))
        self.b_r = np.zeros((1, hidden_size))
        self.b_h = np.zeros((1, hidden_size))

    def forward(self, x, h_prev):
        # Forward pass
        self.z = sigmoid(np.dot(x, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)
        self.r = sigmoid(np.dot(x, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)
        self.h_hat = tanh(np.dot(x, self.W_h) + np.dot(self.r * h_prev, self.U_h) + self.b_h)
        self.h = (1 - self.z) * h_prev + self.z * self.h_hat
        return self.h

# Example Usage
if __name__ == "__main__":
    # MLP example
    mlp = MLP(input_size=10, hidden_size=20, output_size=5)
    x = np.random.randn(3, 10)  # Batch size of 3
    print("MLP output:\n", mlp.forward(x))

    # LSTM example
    lstm = LSTM(input_size=10, hidden_size=20)
    h_prev = np.zeros((3, 20))
    c_prev = np.zeros((3, 20))
    x = np.random.randn(3, 10)  # Batch size of 3
    h, c = lstm.forward(x, h_prev, c_prev)
    print("LSTM output:\n", h)

    # GRU example
    gru = GRU(input_size=10, hidden_size=20)
    h_prev = np.zeros((3, 20))
    x = np.random.randn(3, 10)  # Batch size of 3
    h = gru.forward(x, h_prev)
    print("GRU output:\n", h)
