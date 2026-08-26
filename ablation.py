"""
ablation.py
-----------
Regularization ablation study: trains the same 64-128-64-10 architecture
under three regimes to quantify how dropout + L2 address overfitting:
  1. No regularization (dropout=0, l2=0)
  2. L2 only (dropout=0, l2=1e-4)
  3. Dropout + L2 (dropout=0.3, l2=1e-4)  <- the configuration used in train.py

Produces figures/05_ablation_overfitting.png and outputs/ablation.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_net import DeepNeuralNetwork

RANDOM_STATE = 42

data = load_digits()
X, y = data.data, data.target
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

configs = {
    "No regularization": dict(dropout_rate=0.0, l2_lambda=0.0),
    "L2 only": dict(dropout_rate=0.0, l2_lambda=1e-4),
    "Dropout + L2 (used)": dict(dropout_rate=0.3, l2_lambda=1e-4),
}

results = {}
histories = {}
for name, cfg in configs.items():
    net = DeepNeuralNetwork([64, 128, 64, 10], seed=RANDOM_STATE, **cfg)
    hist = net.fit(X_train, y_train, X_val, y_val, epochs=150, batch_size=32,
                    lr=1e-3, patience=150, verbose=False)  # patience=epochs -> no early stop, see full curve
    test_acc = net.accuracy(X_test, y_test)
    gap = hist["train_acc"][-1] - hist["val_acc"][-1]
    results[name] = {"test_accuracy": float(test_acc),
                      "final_train_acc": float(hist["train_acc"][-1]),
                      "final_val_acc": float(hist["val_acc"][-1]),
                      "train_val_acc_gap": float(gap)}
    histories[name] = hist
    print(f"{name:22s} | test_acc={test_acc:.4f} | "
          f"train_acc={hist['train_acc'][-1]:.4f} val_acc={hist['val_acc'][-1]:.4f} "
          f"| overfit gap={gap:.4f}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = {"No regularization": "#e45756", "L2 only": "#f58518", "Dropout + L2 (used)": "#4c78a8"}
for name, hist in histories.items():
    epochs = range(1, len(hist["val_loss"]) + 1)
    ax.plot(epochs, hist["val_loss"], label=name, color=colors[name])
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation cross-entropy loss")
ax.set_title("Effect of Dropout + L2 regularization on overfitting\n"
              "(lower & flatter validation loss = less overfitting)")
ax.legend()
fig.tight_layout()
fig.savefig("../figures/05_ablation_overfitting.png", bbox_inches="tight")
plt.close(fig)

with open("../outputs/ablation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved ablation plot and JSON.")
