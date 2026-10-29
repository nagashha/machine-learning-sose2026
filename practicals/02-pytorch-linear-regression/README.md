# Practical 2 - Introduction to PyTorch: Linear Regression

**Notebook:** [`practical2_pytorch_linear_regression.ipynb`](practical2_pytorch_linear_regression.ipynb)
**Data:** Superconductivity data set (see [`../data/README.md`](../data/README.md))

The same regression problem as in Practical 1, now solved with the standard PyTorch
workflow. Linear regression is a one-neuron network, so this is the blueprint for every
later deep-learning model.

## What we did

1. **Tensors** - creating tensors, shapes/dtypes, memory sharing with NumPy, moving data
   between CPU and GPU.
2. **Data pipeline** - a custom `Dataset` that normalises features with training-set
   statistics, a 72 / 8 / 20 % split into train / validation / test, and `DataLoader`s
   with shuffling and mini-batches.
3. **Model** - `torch.nn.Linear(81, 1)` with MSE loss.
4. **Training loop** - first with a manual gradient step (`param -= lr * grad` under
   `torch.no_grad()`), then with `torch.optim.SGD`.
5. **Monitoring** - training and validation loss per epoch, plotted with seaborn.
6. **Hyperparameters** - learning rates from 10⁻⁴ to 10, early stopping (patience `N`,
   tolerance `tol`) and batch sizes 4 / 32 / 128 / 1024.

## Key results

- With SGD and a sensible learning rate the model converges to a validation MSE of
  **≈ 340** - the same level as the closed-form OLS solution from Practical 1.
- **Learning rate:** 10⁻⁴-10⁻² converge (slowly to fast); 0.1 and above make the loss
  explode (it overflows to infinity within a few epochs).
- **Early stopping** (`tol = 2`, `N = 5`) stopped after 23 epochs (lr 0.01) and 44 epochs
  (lr 0.001); with lr 0.0001 it ran the full 150 epochs.
- **Batch size:** small batches (4, 32) converge in far fewer epochs because they make
  many more updates per epoch; batch 1024 needed 65 epochs and still had a higher loss.

> **Fixes compared with the submitted version:** the validation set was originally split
> from the *whole* data set, so test rows leaked into training; and the manual gradient
> loop never reset the gradients. Both are fixed and the notebook was re-run.
