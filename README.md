# Machine Learning - Coursework (Summer 2026)

🇹🇷 [Türkçe açıklama için buraya tıklayın → README.tr.md](README.tr.md)

My coursework for the **Machine Learning** course at the University of Göttingen
(summer term 2026), done in a group of three. It goes from linear regression written by
hand to neural networks in PyTorch, and ends with a multimodal price-prediction project
on real Airbnb data.

| # | Folder | Topic | Methods | Result |
|---|---|---|---|---|
| 1 | [`practicals/01-linear-regression`](practicals/01-linear-regression) | Predict the critical temperature of superconductors | OLS and ridge regression from scratch (NumPy), cross-validation | R² 0.52 → 0.73 with more data |
| 2 | [`practicals/02-pytorch-linear-regression`](practicals/02-pytorch-linear-regression) | Same task, now in PyTorch | Tensors, `Dataset`/`DataLoader`, SGD, learning rate & batch size, early stopping | Val. MSE ≈ 340 (same as closed-form OLS) |
| 3 | [`practicals/03-image-classification-cifar10`](practicals/03-image-classification-cifar10) | Classify CIFAR-10 images | Logistic regression, MLP, L2 regularisation, LeNet-5 CNN | 39 % → 67 % accuracy |
| ★ | [`airbnb-price-prediction`](airbnb-price-prediction) | **Final project:** nightly prices of Airbnb listings in Dublin | Tabular + text + spatial features, Ridge / Random Forest / Gradient Boosting, tuning | MAE €41/night vs. €94 for the baseline, R² 0.72 |

## Repository structure

```
.
├── practicals/
│   ├── data/                              # superconductivity CSV goes here (not in git)
│   ├── 01-linear-regression/              # notebook + short README
│   ├── 02-pytorch-linear-regression/
│   └── 03-image-classification-cifar10/
├── airbnb-price-prediction/               # final project (scripts, not a notebook)
│   ├── src/                               # data_prep → eda → train → tune → evaluate
│   ├── data/                              # sample data in git, full data downloaded
│   └── outputs/                           # figures and metric tables
└── requirements.txt
```

## Getting started

```bash
git clone <this-repo>
cd machine-learning-sose2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **Practicals 1-2:** download the superconductivity data as described in
  [`practicals/data/README.md`](practicals/data/README.md), then open the notebooks with
  `jupyter lab`.
- **Practical 3:** CIFAR-10 is downloaded automatically; a GPU (e.g. Kaggle or Colab) is
  recommended.
- **Airbnb project:** see [`airbnb-price-prediction/README.md`](airbnb-price-prediction/README.md).
  A small sample is included, so `AIRBNB_RAW_CSV=data/sample/listings_sample.csv` lets you
  test the whole pipeline in seconds.

## Team

Muhammed Nagaş · Felix Fischer · Niclas Langer

## Acknowledgements

The practical notebooks are based on exercise templates provided by the course
instructors (task descriptions and some helper code); the solutions and answers are ours.
Data: [UCI Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data),
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
[Inside Airbnb](https://insideairbnb.com/) (CC BY 4.0).
