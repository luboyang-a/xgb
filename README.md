# ⚡ XGB-Lightning (Zero-Copy & Dual-Layout Edition)

> **Branch:** `feat/lightning-zero-copy`  
> **Benchmark Performance:** **235.55s** (Fashion-MNIST 60K Samples)

This branch represents the high-performance core of the XGB implementation. By restructuring the **Memory Layout** and maximizing **CPU Cache Hit Rates**, we have compressed the training time from 600s+ down to the **230s range**, achieving a **2.5x speedup**.

---

## 🚀 Key Technical Innovations

### 1. Dual-Layout Memory Hedging
The most critical optimization in this project. We identified that GBDT has diametrically opposed memory access patterns during different stages, so we implemented a "Switchable Layout" strategy:

* **Training Phase (Fit) -> Fortran-order (Column-major)** Histogram construction requires scanning features (columns) across all samples. Using a column-major layout ensures that data for the same feature is physically contiguous in memory.  
  **Result:** Maximizes CPU **hardware prefetching** efficiency and eliminates memory latency during feature binning.
  
* **Inference Phase (Predict) -> C-order (Row-major)** Predicting a single sample requires looking up different features (rows). A row-major layout ensures all features for a single sample are stored together.  
  **Result:** Achieves optimal **L1/L2 Cache locality** during tree traversal.



### 2. Zero-Copy Indexing & Partitioning
* **In-place Partitioning**: We manage samples via an `int[:] indices` array at the Cython level. Splitting is achieved by swapping indices rather than physically moving large chunks of data in memory.
* **Typed Memoryviews**: By using Cython memoryviews, we bypass Python object overhead entirely, reaching raw C execution speeds.

### 3. Layout-Agnostic Cython Interface
* By defining the Cython interface as `unsigned char[:, :]` (generic 2D view), we decoupled the layout restrictions. This allows the engine to read both F-order and C-order data with "zero-copy" overhead, avoiding expensive intermediate memory allocations.

---

## 📊 Benchmark Results (Fashion-MNIST)

| Step | Time (s) | Status |
| :--- | :--- | :--- |
| **Data Loading** | ~2.0s | NumPy Native |
| **Binning (Histogram)** | 4.58s | Minimal Preprocessing |
| **Training (100 Trees)** | **235.55s** | **Lightning Engine Active** |
| **Accuracy** | **87.12%** | High Precision Maintained |



---

## 🛠️ Getting Started

1.  **Environment**: Ensure `Cython` and `numpy` are installed.
2.  **Compile the Engine**:
    ```bash
    python setup.py build_ext --inplace
    ```
3.  **Run the Benchmark**:
    ```bash
    python main.py
    ```

---

## 🏁 Conclusion
**XGB-Lightning** proves that in high-performance computing, understanding **Memory Locality** is often more effective than simply throwing more threads at a problem. This implementation breaks the "Memory Wall" and delivers parallel-grade performance on a single-threaded execution core.