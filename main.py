import gzip
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
from boost import OneVsRestBoost
from xgb_utils import utils 

def load_mnist_local(images_path, labels_path):
    print(f"Loading: {images_path}...")
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

def run_benchmark():
    try:
        X_train, y_train = load_mnist_local('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
        X_test, y_test = load_mnist_local('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    print(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")

    print("Binning data (Histogram-based)...")
    start_bin = time.time()
    X_train_bin, bin_info = utils.equal_frequency(X_train, n_bin=256)
    X_test_bin = utils.transform_equal_frequency(X_test, bin_info, n_bin=256)
    
    X_train_bin = X_train_bin.astype(np.uint8)
    X_test_bin = X_test_bin.astype(np.uint8)
    print(f"Binning cost: {time.time() - start_bin:.2f}s")

    params = {
        'n_estimators': 50,
        'max_depth': 6,
        'learning_rate': 0.1,
        'l2_reg': 1.0,
        'gamma': 0.1
    }

    print("Starting training (Cython engine)...")
    start_train = time.time()
    ovr = OneVsRestBoost(n_classes=10, **params)
    ovr.fit(X_train_bin, y_train)
    print(f"Training completed. Total time: {time.time() - start_train:.2f}s")

    print("\nGenerating Report...")
    y_pred = ovr.predict(X_test_bin)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    target_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    run_benchmark()