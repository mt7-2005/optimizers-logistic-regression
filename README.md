# Logistic Regression Optimizers — GD, SGD, Mini-Batch GD, RMSprop, Adam

Five Jupyter notebooks that train a **binary-classification Logistic Regression** model with five different optimizers.
The model, the loss (Binary Cross-Entropy), all five optimizers and all metrics are written **from scratch in NumPy**;
each notebook also checks the hand-written metrics against `sklearn.metrics`.

Companion to [optimizers-regression](https://github.com/mt7-2005/optimizers-regression) (same optimizers on a regression task).

## Dataset — Breast Cancer Wisconsin

- 569 samples, 30 numeric features computed from cell-nucleus images
- Target: `1` = **Malignant** (positive class), `0` = Benign
- Built into scikit-learn (`load_breast_cancer`), so no download or internet is needed

Preprocessing: stratified 80/20 train/test split → standardization.

## Metrics

Accuracy, Precision, Recall, F1 (from the confusion matrix) and Loss (Binary Cross-Entropy).

## Notebooks

| # | Notebook | Optimizer |
|---|---|---|
| 1 | [01_Batch_Gradient_Descent.ipynb](01_Batch_Gradient_Descent.ipynb) | Batch Gradient Descent |
| 2 | [02_Stochastic_Gradient_Descent.ipynb](02_Stochastic_Gradient_Descent.ipynb) | Stochastic Gradient Descent (SGD) |
| 3 | [03_MiniBatch_Gradient_Descent.ipynb](03_MiniBatch_Gradient_Descent.ipynb) | Mini-Batch Gradient Descent |
| 4 | [04_RMSprop.ipynb](04_RMSprop.ipynb) | RMSprop |
| 5 | [05_Adam.ipynb](05_Adam.ipynb) | Adam |

Each notebook contains: dataset + EDA, preprocessing, the model and gradients, metric functions, a sklearn baseline,
the optimizer implementation, results table, loss/accuracy curves, confusion matrix and a learning-rate experiment.

## Results (test set, 114 samples)

| Optimizer | Settings | Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| sklearn `LogisticRegression` | lbfgs, L2 | 0.0773 | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| Batch GD | lr=0.1, 500 epochs | 0.0737 | 0.9825 | 1.0000 | 0.9524 | 0.9756 |
| SGD | lr=0.01, 50 epochs | 0.0717 | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| Mini-Batch GD | lr=0.05, batch 32, 100 epochs | 0.0712 | 0.9825 | 1.0000 | 0.9524 | 0.9756 |
| RMSprop | lr=0.001, batch 32, 100 epochs | 0.0676 | 0.9825 | 1.0000 | 0.9524 | 0.9756 |
| Adam | lr=0.001, batch 32, 100 epochs | 0.0873 | 0.9825 | 1.0000 | 0.9524 | 0.9756 |

## How to run

**Google Colab:** File → Open notebook → GitHub tab → `mt7-2005/optimizers-logistic-regression`, then Runtime → Run all.

**Kaggle:** New Notebook → File → Import Notebook, then Run All (no internet needed).

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook
```
