# Data for Practicals 1 and 2

Both notebooks use the **Superconductivity Data Set** (UCI Machine Learning Repository):
21,263 superconducting materials, 81 physical/chemical features and the target
`critical_temp` (the temperature below which the material becomes superconducting).

The CSV (~24 MB) is not stored in the repository. To run the notebooks:

1. Download the data set from <https://archive.ics.uci.edu/dataset/464/superconductivty+data>
2. Unzip it and put `train.csv` here, renamed to **`superconduct_train.csv`**:

```
practicals/data/superconduct_train.csv
```

Source: Hamidieh, K. (2018). *A data-driven statistical model for predicting the critical
temperature of a superconductor.* Computational Materials Science, 154, 346-354.
License: CC BY 4.0.

Practical 3 uses CIFAR-10, which `torchvision` downloads automatically.
