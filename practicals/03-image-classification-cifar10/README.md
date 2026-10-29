# Practical 3 - Image Classification on CIFAR-10

**Notebook:** [`practical3_cifar10_classification.ipynb`](practical3_cifar10_classification.ipynb)
**Data:** CIFAR-10 (60,000 colour images, 32x32 px, 10 classes) - downloaded automatically by `torchvision`

Compare three model families of increasing power on the same image task, all trained in
PyTorch with Adam (lr = 0.001), cross-entropy loss, batch size 256 and random horizontal
flips as data augmentation.

## Models

| Model | Idea | Parameters |
|---|---|---|
| Logistic regression | one linear layer on the flattened 3,072 pixels | ~31 k |
| MLP | 3072 → 2048 → 512 → 128 → 10 with ReLU | ~7.4 M |
| MLP + L2 | same MLP with weight decay 5e-5 | ~7.4 M |
| LeNet-5 | 2 conv + pooling blocks, 3 fully-connected layers, ReLU | ~62 k |

## Results (test set, 10,000 images)

| Model | Test accuracy |
|---|---|
| Logistic regression | ≈ 39 % |
| MLP | ≈ 56 % |
| MLP + L2 | ≈ 56 % |
| **LeNet-5** | **≈ 67 %** |

**Take-aways:** the CNN wins with ~100x fewer parameters than the MLP, because
convolutions share weights and exploit local structure in images. The MLP overfits
(training accuracy keeps rising, test accuracy plateaus around 56 %). The LeNet-5
confusion matrix shows the typical mix-ups: cat ↔ dog, automobile ↔ truck, bird ↔ deer.

> **Note:** the notebook was run on a Kaggle GPU. The regularised MLP was trained in two
> consecutive calls (40 epochs in total), which is marked in the notebook.
