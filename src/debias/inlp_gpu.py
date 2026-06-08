import numpy as np
import cupy as cp
from cuml.svm import LinearSVC as cuLinearSVC
from cuml.metrics import accuracy_score
from typing import List
from tqdm import tqdm

class GPUClassifier:
    def __init__(self, **kwargs):
        self.model = cuLinearSVC(**kwargs)

    def train_network(self, X_train, Y_train, X_dev, Y_dev):
        """Train classifier and return dev set accuracy.
        
        Note: Input arrays should be cupy arrays.
        """
        self.model.fit(X_train, Y_train)
        predictions = self.model.predict(X_dev)
        score = accuracy_score(Y_dev, predictions)
        return float(score)  # Convert from device array to float

    def get_weights(self):
        """Get the classifier weights."""
        w = self.model.coef_
        if len(w.shape) == 1:
            w = cp.expand_dims(w, 0)
        return w

def get_rowspace_projection(W: cp.ndarray) -> cp.ndarray:
    """Get rowspace projection matrix of W."""
    if cp.allclose(W, 0):
        w_basis = cp.zeros_like(W.T)
    else:
        # QR decomposition for orthogonal basis (equivalent to scipy.linalg.orth)
        q, _ = cp.linalg.qr(W.T)
        w_basis = q[:, :W.shape[0]]  # Only keep the first rank(W) columns
    return w_basis.dot(w_basis.T)

def get_projection_to_intersection_of_nullspaces(rowspace_projections: List[cp.ndarray], input_dim: int):
    """Get projection to intersection of nullspaces."""
    I = cp.eye(input_dim)
    Q = cp.sum(rowspace_projections, axis=0)
    return I - get_rowspace_projection(Q)

def get_debiasing_projection(X_train, Y_train, X_dev, Y_dev, input_dim, n_classifiers, dropout_rate=0.0, min_accuracy=0.0):
    """Compute INLP debiasing projection matrix on GPU.
    
    Args:
        X_train: Training vectors (numpy array)
        Y_train: Training labels (numpy array)
        X_dev: Development set vectors (numpy array)
        Y_dev: Development set labels (numpy array)
        input_dim: Dimensionality of input vectors
        n_classifiers: Number of classifiers to train
        dropout_rate: Rate of dropout to apply during training (default: 0.0)
        min_accuracy: Minimum accuracy threshold for keeping a classifier
        
    Returns:
        numpy.ndarray: The final projection matrix
    """
    # Transfer data to GPU
    X_train_gpu = cp.array(X_train)
    Y_train_gpu = cp.array(Y_train)
    X_dev_gpu = cp.array(X_dev)
    Y_dev_gpu = cp.array(Y_dev)
    
    I = cp.eye(input_dim)
    rowspace_projections = []
    
    X_train_cp = X_train_gpu.copy()
    X_dev_cp = X_dev_gpu.copy()
    
    cls_params = {
        "fit_intercept": False,
        "output_type": "cupy",  # Ensure cuML outputs cupy arrays
    }

    pbar = tqdm(range(n_classifiers), desc="Training classifiers")
    for i in pbar:
        if dropout_rate > 0:
            # Create dropout mask
            dropout_scale = 1.0 / (1 - dropout_rate + 1e-6)
            dropout_mask = (cp.random.rand(*X_train_cp.shape) < (1 - dropout_rate)).astype(cp.float32) * dropout_scale
            # Apply dropout to training data
            X_train_masked = X_train_cp * dropout_mask
        else:
            X_train_masked = X_train_cp
        
        clf = GPUClassifier(**cls_params)
        acc = clf.train_network(X_train_masked, Y_train_gpu, X_dev_cp, Y_dev_gpu)
        pbar.set_description(f"iteration: {i}, accuracy: {acc:.3f}")
        
        if acc < min_accuracy:
            continue
            
        W = clf.get_weights()
        P_rowspace_wi = get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace_wi)
        
        # Project to intersection of nullspaces
        P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
        X_train_cp = cp.dot(P, X_train_gpu.T).T
        X_dev_cp = cp.dot(P, X_dev_gpu.T).T

    # Final projection matrix
    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    
    # Transfer final projection matrix back to CPU
    return cp.asnumpy(P)

# Optional: Memory management utility
def clear_gpu_memory():
    """Clear GPU memory cache."""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()