# Deep Learning Application in Data Science — Week 5

Handwritten digit classification (UCI Optical Recognition of Handwritten Digits,
via `sklearn.datasets.load_digits`) using a from-scratch NumPy deep neural network,
benchmarked against Logistic Regression and scikit-learn's MLPClassifier.

## Why NumPy instead of TensorFlow/PyTorch
This project was built in a network-isolated sandbox where TensorFlow/PyTorch
could not be installed. `neural_net.py` implements the forward pass, backprop,
and Adam optimizer from scratch, replicating what those frameworks do internally.

## Files
- `neural_net.py` — DenseLayer + DeepNeuralNetwork classes (forward/backward/Adam/dropout/L2)
- `train.py` — loads data, trains the custom DNN + two baselines, generates all figures/metrics
- `ablation.py` — regularization ablation study (no-reg vs L2-only vs dropout+L2)

## Run
```bash
pip install numpy scikit-learn matplotlib
cd code_bundle
python train.py
python ablation.py
```

Outputs land in `../figures` and `../outputs` (metrics.json, classification_reports.txt).

## Results
| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Custom Deep Neural Network (64-128-64-10) | 98.52% | 0.9848 |
| Logistic Regression | 98.52% | 0.9852 |
| scikit-learn MLPClassifier | 97.41% | 0.9739 |

See the full report (`Week5_Deep_Learning_Report.docx`) for architecture
diagrams, training curves, confusion matrices, and the overfitting ablation.
