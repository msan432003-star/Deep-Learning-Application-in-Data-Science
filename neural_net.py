"""
neural_net.py
--------------
A minimal, dependency-light (NumPy-only) feed-forward deep neural network
implementing the same core mechanics that TensorFlow/PyTorch provide under
the hood: dense (fully connected) layers, ReLU / Softmax activations,
mini-batch gradient descent with the Adam optimizer, dropout regularization,
L2 weight decay, and early stopping.

Why a from-scratch implementation?
This project was built in an offline, network-restricted sandbox where
`pip install tensorflow` / `pip install torch` could not reach PyPI. Rather
than skip the deep learning requirement, the network (forward pass,
backpropagation, and Adam update rule) is implemented directly on top of
NumPy. This is functionally equivalent to a `Sequential` model of
Dense -> ReLU -> Dropout -> Dense -> ReLU -> Dropout -> Dense -> Softmax
in Keras/PyTorch, and is cross-checked against scikit-learn's MLPClassifier
(see train_baseline.py) as an independent sanity check on the results.

Author: Deep Learning Internship - Week 5
"""

import numpy as np


def one_hot(y, num_classes):
    m = np.zeros((y.shape[0], num_classes))
    m[np.arange(y.shape[0]), y] = 1.0
    return m


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(probs, y_onehot, weights=None, l2_lambda=0.0):
    m = y_onehot.shape[0]
    eps = 1e-12
    data_loss = -np.sum(y_onehot * np.log(probs + eps)) / m
    reg_loss = 0.0
    if weights is not None and l2_lambda > 0:
        reg_loss = (l2_lambda / (2 * m)) * sum(np.sum(w ** 2) for w in weights)
    return data_loss + reg_loss


class DenseLayer:
    """A single fully connected layer with He initialization."""

    def __init__(self, n_in, n_out, rng):
        # He initialization: suited for ReLU activations, keeps activation
        # variance roughly constant across layers -> faster/more stable training.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))

        # Adam optimizer moment buffers
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dz):
        m = self.x.shape[0]
        self.dW = self.x.T @ dz / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        dx = dz @ self.W.T
        return dx

    def adam_step(self, lr, l2_lambda, m_samples, t, beta1=0.9, beta2=0.999, eps=1e-8):
        dW = self.dW + (l2_lambda / m_samples) * self.W  # L2 weight decay
        db = self.db

        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * (dW ** 2)
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db ** 2)

        mW_hat = self.mW / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        mb_hat = self.mb / (1 - beta1 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class DeepNeuralNetwork:
    """
    Fully connected feed-forward network:
        Input -> [Dense -> ReLU -> Dropout] x (L-1) -> Dense -> Softmax

    Trained with mini-batch gradient descent + Adam, L2 regularization,
    dropout, and early stopping on validation loss.
    """

    def __init__(self, layer_sizes, dropout_rate=0.3, l2_lambda=1e-4, seed=42):
        self.rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.layers = [
            DenseLayer(layer_sizes[i], layer_sizes[i + 1], self.rng)
            for i in range(len(layer_sizes) - 1)
        ]
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _forward(self, X, training):
        activations = X
        self._cache_dropout_masks = []
        self._cache_z = []
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward(activations)
            a = relu(z)
            if training and self.dropout_rate > 0:
                mask = (self.rng.random(a.shape) > self.dropout_rate).astype(a.dtype)
                a = a * mask / (1 - self.dropout_rate)  # inverted dropout
            else:
                mask = np.ones_like(a)
            self._cache_dropout_masks.append(mask)
            self._cache_z.append(z)
            activations = a
        z_out = self.layers[-1].forward(activations)
        probs = softmax(z_out)
        return probs

    def _backward(self, probs, y_onehot):
        m = y_onehot.shape[0]
        dz = (probs - y_onehot)  # softmax + cross-entropy gradient
        dx = self.layers[-1].backward(dz)
        for i in reversed(range(len(self.layers) - 1)):
            mask = self._cache_dropout_masks[i]
            dx = dx * mask / (1 - self.dropout_rate) if self.dropout_rate > 0 else dx
            dz = dx * relu_grad(self._cache_z[i])
            dx = self.layers[i].backward(dz)

    def predict_proba(self, X):
        return self._forward(X, training=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32,
            lr=1e-3, patience=20, verbose=True):
        num_classes = self.layer_sizes[-1]
        y_train_oh = one_hot(y_train, num_classes)
        n = X_train.shape[0]
        t = 0
        best_val_loss = np.inf
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train_oh[perm]

            for start in range(0, n, batch_size):
                end = start + batch_size
                xb, yb = X_shuf[start:end], y_shuf[start:end]
                probs = self._forward(xb, training=True)
                self._backward(probs, yb)
                t += 1
                for layer in self.layers:
                    layer.adam_step(lr, self.l2_lambda, xb.shape[0], t)

            # ---- epoch-end evaluation ----
            train_probs = self._forward(X_train, training=False)
            val_probs = self._forward(X_val, training=False)
            weights = [l.W for l in self.layers]
            train_loss = cross_entropy_loss(train_probs, y_train_oh, weights, self.l2_lambda)
            val_loss = cross_entropy_loss(val_probs, one_hot(y_val, num_classes))
            train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train)
            val_acc = np.mean(np.argmax(val_probs, axis=1) == y_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

            # ---- early stopping ----
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = [(l.W.copy(), l.b.copy()) for l in self.layers]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no val_loss improvement for {patience} epochs).")
                    break

        if best_state is not None:
            for layer, (W, b) in zip(self.layers, best_state):
                layer.W, layer.b = W, b

        return self.history
