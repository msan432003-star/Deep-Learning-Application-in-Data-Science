"""
train.py
--------
End-to-end experiment for Week 5: Deep Learning Application in Data Science.

Dataset : sklearn's `load_digits` -- a copy of the UCI "Optical Recognition
          of Handwritten Digits" dataset (1,797 8x8 grayscale images, 10
          balanced classes, digits 0-9).
Task    : Multi-class image classification.
Model   : A from-scratch feed-forward deep neural network (see neural_net.py)
          64 -> 128 -> 64 -> 10, ReLU + Dropout + Softmax, trained with
          mini-batch Adam, L2 regularization and early stopping.
Baselines: (1) Logistic Regression  (2) scikit-learn MLPClassifier
          -- included so the custom implementation's results can be
          cross-checked against an independent, well-tested library.

Outputs -> ../figures/*.png and ../outputs/metrics.json
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

from neural_net import DeepNeuralNetwork

FIG_DIR = "../figures"
OUT_DIR = "../outputs"
RANDOM_STATE = 42

plt.rcParams.update({"figure.dpi": 140, "font.size": 10})


def load_and_split():
    data = load_digits()
    X, y = data.data, data.target

    # 70% train / 15% val / 15% test, stratified to preserve class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
            X_train, X_test, data)


def plot_sample_digits(data):
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.4))
    for i, ax in enumerate(axes.flat):
        idx = np.where(data.target == i)[0][0]
        ax.imshow(data.images[idx], cmap="gray_r")
        ax.set_title(f"label={i}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample images from the Optical Handwritten Digits dataset")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/00_sample_digits.png", bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(data):
    counts = np.bincount(data.target)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(10), counts, color="#3b6fa0")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Number of samples")
    ax.set_title("Class distribution (balanced across 10 digits)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_class_balance.png", bbox_inches="tight")
    plt.close(fig)


def plot_architecture(layer_sizes):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    v_gap = 0.055
    x_positions = np.linspace(0.08, 0.92, len(layer_sizes))
    max_draw = 14  # cap neurons drawn per layer for legibility
    layer_labels = ["Input\n(64 px features)", "Hidden 1\n(128, ReLU+Dropout)",
                     "Hidden 2\n(64, ReLU+Dropout)", "Output\n(10, Softmax)"]

    positions = []
    for li, size in enumerate(layer_sizes):
        n_draw = min(size, max_draw)
        ys = np.linspace(0.15, 0.85, n_draw)
        positions.append([(x_positions[li], y) for y in ys])

    # edges
    for li in range(len(positions) - 1):
        for (x1, y1) in positions[li]:
            for (x2, y2) in positions[li + 1]:
                ax.plot([x1, x2], [y1, y2], color="#c7d3e0", linewidth=0.4, zorder=1)

    colors = ["#4c78a8", "#f58518", "#f58518", "#54a24b"]
    for li, pts in enumerate(positions):
        for (x, y) in pts:
            ax.scatter([x], [y], s=180, color=colors[li], edgecolor="white",
                       zorder=2, linewidth=0.8)
        if layer_sizes[li] > max_draw:
            ax.text(x_positions[li], 0.06, f"... {layer_sizes[li]} units",
                    ha="center", fontsize=8, color="#555")
        ax.text(x_positions[li], 0.97, layer_labels[li], ha="center",
                fontsize=9, fontweight="bold")

    ax.set_title("Network Architecture: 64 -> 128 -> 64 -> 10 "
                  "(Dense + ReLU + Dropout, Softmax output)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_architecture.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, fname_prefix="03"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{fname_prefix}_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{fname}", bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_misclassified(X_test_raw, y_test, y_pred, n=10):
    wrong_idx = np.where(y_test != y_pred)[0]
    n = min(n, len(wrong_idx))
    if n == 0:
        return
    fig, axes = plt.subplots(2, 5, figsize=(9, 3.8))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        idx = wrong_idx[i]
        ax.imshow(X_test_raw[idx].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"true={y_test[idx]} pred={y_pred[idx]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Examples of misclassified digits (custom DNN, test set)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_misclassified.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(names, accs, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model comparison on held-out test set")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.003, f"{a:.3f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=3)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Macro-P: {prec:.4f} | Macro-R: {rec:.4f} | Macro-F1: {f1:.4f}")
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec,
            "f1_macro": f1, "report": report}


def main():
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_train_raw, X_test_raw, data) = load_and_split()

    print(f"Train/Val/Test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")

    plot_sample_digits(data)
    plot_class_balance(data)
    plot_architecture([64, 128, 64, 10])

    # ---------------- Custom from-scratch deep neural network ----------------
    print("\nTraining custom NumPy deep neural network (64-128-64-10)...")
    t0 = time.time()
    dnn = DeepNeuralNetwork([64, 128, 64, 10], dropout_rate=0.3, l2_lambda=1e-4, seed=RANDOM_STATE)
    history = dnn.fit(X_train, y_train, X_val, y_val, epochs=300, batch_size=32,
                       lr=1e-3, patience=25, verbose=True)
    dnn_time = time.time() - t0
    print(f"Custom DNN training time: {dnn_time:.2f}s over {len(history['train_loss'])} epochs")

    plot_training_curves(history)
    y_pred_dnn = dnn.predict(X_test)
    cm_dnn = plot_confusion(y_test, y_pred_dnn, "Confusion Matrix - Custom Deep Neural Network",
                             "04_confusion_dnn.png")
    plot_misclassified(X_test_raw, y_test, y_pred_dnn)
    dnn_metrics = evaluate("Custom Deep Neural Network (64-128-64-10)", y_test, y_pred_dnn)
    dnn_metrics["train_time_sec"] = dnn_time
    dnn_metrics["epochs_run"] = len(history["train_loss"])

    # ---------------- Baseline 1: Logistic Regression -------------------
    print("\nTraining Logistic Regression baseline...")
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    lr_metrics = evaluate("Logistic Regression (linear baseline)", y_test, y_pred_lr)

    # ---------------- Baseline 2: scikit-learn MLPClassifier -------------
    print("\nTraining scikit-learn MLPClassifier (library-based neural net)...")
    t0 = time.time()
    sk_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                            solver="adam", alpha=1e-4, batch_size=32,
                            learning_rate_init=1e-3, max_iter=300,
                            early_stopping=True, n_iter_no_change=25,
                            validation_fraction=0.15, random_state=RANDOM_STATE)
    sk_mlp.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    sk_time = time.time() - t0
    y_pred_sk = sk_mlp.predict(X_test)
    sk_metrics = evaluate("scikit-learn MLPClassifier (128,64)", y_test, y_pred_sk)
    sk_metrics["train_time_sec"] = sk_time

    results = {
        "Custom DNN\n(NumPy)": dnn_metrics,
        "Logistic\nRegression": lr_metrics,
        "sklearn\nMLPClassifier": sk_metrics,
    }
    plot_model_comparison(results)

    # ---------------- Save everything ----------------
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "dataset": {
                "name": "Optical Recognition of Handwritten Digits (sklearn load_digits)",
                "n_samples": int(data.data.shape[0]),
                "n_features": int(data.data.shape[1]),
                "n_classes": 10,
                "train_size": int(X_train.shape[0]),
                "val_size": int(X_val.shape[0]),
                "test_size": int(X_test.shape[0]),
            },
            "custom_dnn": {k: v for k, v in dnn_metrics.items() if k != "report"},
            "logistic_regression": {k: v for k, v in lr_metrics.items() if k != "report"},
            "sklearn_mlp": {k: v for k, v in sk_metrics.items() if k != "report"},
        }, f, indent=2)

    with open(f"{OUT_DIR}/classification_reports.txt", "w") as f:
        f.write("=== Custom Deep Neural Network ===\n")
        f.write(dnn_metrics["report"] + "\n\n")
        f.write("=== Logistic Regression ===\n")
        f.write(lr_metrics["report"] + "\n\n")
        f.write("=== scikit-learn MLPClassifier ===\n")
        f.write(sk_metrics["report"] + "\n")

    print("\nAll figures saved to", FIG_DIR)
    print("Metrics saved to", OUT_DIR)


if __name__ == "__main__":
    main()
