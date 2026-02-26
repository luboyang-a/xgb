# MyXGBoost

A lightweight Gradient Boosting Decision Tree (GBDT) implementation featuring **Cython** acceleration and **Histogram-based** binning.

## ✨ Features
- **Cython Engine**: Core splitting logic is implemented in Cython to bypass Python's global interpreter lock and slow loops.
- **Histogram Binning**: Features are compressed into discrete bins (Default: 256) to drastically reduce computation cost for high-dimensional data.
- **One-vs-Rest (OvR)**: Native support for multi-class classification tasks.

## 📊 Benchmark: Fashion-MNIST
The model achieved the following performance on the official Fashion-MNIST dataset:
- **Training Set**: 60,000 | **Test Set**: 10,000
- **Feature Dimension**: 784 (Pixel intensities)
- **Accuracy**: **87.15%**

### Classification Detail (F1-Score)
| Class | F1-Score |
| :--- | :--- |
| Trouser | 0.98 |
| Sandal | 0.96 |
| Bag | 0.96 |
| Ankle boot | 0.95 |
| Sneaker | 0.94 |
| **Overall Accuracy** | **87.15%** |

## 🛠️ Installation & Usage
1. **Compile the Cython core**:
   ```bash
   python setup.py build_ext --inplace