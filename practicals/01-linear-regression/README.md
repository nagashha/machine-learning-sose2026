# Practical 1 - Linear Regression

**Notebook:** [`practical1_linear_regression.ipynb`](practical1_linear_regression.ipynb)
**Data:** Superconductivity data set (see [`../data/README.md`](../data/README.md))

Predict the **critical temperature** of a superconductor from 81 material features, with
linear models written by hand in NumPy and then with scikit-learn.

## What we did

1. **Exploration** - histogram of the target and a 9x9 grid of scatter plots (target vs.
   every feature); z-score normalisation using training-set statistics only.
2. **OLS by hand** - closed-form solution `w = (XᵀX)⁻¹ Xᵀy` with an intercept column.
3. **Same model with scikit-learn** (`LinearRegression`) to check our implementation, then
   trained on the full training set instead of 200 samples.
4. **Ridge regression by hand** - `w = (XᵀX + αI)⁻¹ Xᵀy` (intercept not penalised), swept
   over α from 10⁻⁷ to 10⁷: coefficient paths and test error vs. α. Repeated with
   `sklearn.linear_model.Ridge`.
5. **10-fold cross-validation** for α, written from scratch, on the small and the full
   training set.

## Key results (test set, 500 materials)

| Model | Training data | MSE | R² |
|---|---|---|---|
| OLS (own implementation) | 200 samples | 599.7 | 0.52 |
| OLS (scikit-learn) - identical | 200 samples | 599.7 | 0.52 |
| Ridge, α = 10 | 200 samples | 423.1 | 0.66 |
| OLS (scikit-learn) | 20,763 samples | 341.5 | 0.73 |

**Take-aways:** with only 200 samples and 81 features, OLS overfits and regularisation
helps a lot. With the full training set the optimal α becomes tiny, because plenty of
data already keeps the weights stable. Cross-validation finds roughly the same α as the
test set, without touching the test data.
