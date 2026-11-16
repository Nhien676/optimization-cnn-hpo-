# Hyperparameter Optimization for CNN on CIFAR-10 using Optuna

## 1. Problem Statement
Manual hyperparameter tuning for Convolutional Neural Networks (CNNs) is time-consuming and prone to suboptimal local minima. This project automates hyperparameter search for a `SimpleCNN` image classifier on the CIFAR-10 dataset.

## 2. Problem Formulation
- **Decision Variables ($x$):** `lr` $\in [10^{-5}, 10^{-2}]$ (log-uniform), `batch_size` $\in \{16, 32, 64\}$, `optimizer` $\in \{\text{Adam}, \text{RMSprop}, \text{SGD}\}$, `dropout_rate` $\in [0.1, 0.7]$ (uniform).
- **Objective ($f(x)$):** Maximize Validation Accuracy after 10 epochs.
- **Constraints / Budget:** 30 Optuna trials $\times$ 10 epochs; 80:20 train/validation split (40k/10k) with 10k test images held out independently.

## 3. Method
- **Dataset:** CIFAR-10 Public Benchmark (Alex Krizhevsky, 2009).
- **Baseline:** Default configuration (`lr=0.001`, `batch_size=32`, `Adam`, `dropout=0.5`).
- **Optimization Algorithm:** Tree-structured Parzen Estimator (TPE) via **Optuna** (Python 3.11, PyTorch, seed=42).

## 4. Results
| Configuration | Validation Acc | Test Acc (Held-out) | Generalization Gap | Hyperparameters |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Default)** | 68.48% | **69.26%** | **+0.78%** | `lr=0.001, batch=32, Adam, dropout=0.5` |
| **Best Trial (#23)** | **70.37%** | 57.38% | -12.99% | `lr=0.002025, batch=16, SGD, dropout=0.2531` |
| **Difference (TPE - Base)** | **+1.89%** | -11.88% | — | Optuna Bayesian TPE (30 trials) |

- **Key Findings:** Across 30 trials, validation accuracy ranges from 34.03% (min) to 70.37% (max) with a median of 66.45%. While Trial #23 improves validation accuracy by +1.89%, held-out test evaluation reveals validation overfitting (-12.99% generalization gap) caused by lower regularization (`dropout=0.2531`) on a single seed. Consequently, the `Baseline` configuration (`Adam`, `dropout=0.5`) is selected as the safer model for deployment. All logs and plots (`optimization_history.png`, `param_importances.png`) are available in `notebooks/results/` and `notebooks/figures/`.

## 5. How to Run
```bash
git clone [https://github.com/Nhien676/optimization-cnn-hpo-.git](https://github.com/Nhien676/optimization-cnn-hpo-.git)
cd optimization-cnn-hpo-
pip install -r requirements.txt
jupyter notebook notebooks/cnn_hpo.ipynb  
