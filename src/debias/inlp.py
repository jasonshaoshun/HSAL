import numpy as np
import scipy
from typing import List
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import multiprocessing

# an abstract class for linear classifiers

class Classifier(object):

    def __init__(self):

        pass

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """

        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError




class SKlearnClassifier(Classifier):

    def __init__(self, m):

        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w




def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) # orthogonal basis

    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)

    return P

def get_debiasing_projection(X_train, Y_train, X_dev, Y_dev, input_dim, n_classifiers, dropout_rate=0.1, min_accuracy=0.0):
    """Compute INLP debiasing projection matrix with optimized implementation.
    
    Args:
        X_train: Training vectors
        Y_train: Training labels
        X_dev: Development set vectors
        Y_dev: Development set labels
        input_dim: Dimensionality of input vectors
        n_classifiers: Number of classifiers to train
        dropout_rate: Rate of dropout to apply during training (default: 0.1)
        min_accuracy: Minimum accuracy threshold for keeping a classifier
    """
    rowspace_projections = []
    n_samples = X_train.shape[0]
    
    # Pre-compute projection matrix
    P = np.eye(input_dim)
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    Ws = []

    cls_params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}
    # cls_params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': multiprocessing.cpu_count()}
    
    # Initialize progress bar
    with tqdm(range(n_classifiers), desc="Training classifiers") as pbar:
        for i in pbar:
            # Efficient dropout implementation
            dropout_scale = 1./(1 - dropout_rate + 1e-6)
            dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale


            # Train classifier
            clf = SKlearnClassifier(LinearSVC(**cls_params))
            # clf = SKlearnClassifier(SGDClassifier(**cls_params)
            acc = clf.train_network((X_train_cp * dropout_mask), Y_train, X_dev_cp, Y_dev)
            pbar.set_description(f"iteration: {i}, accuracy: {acc:.3f}")
            
            if acc < min_accuracy:
                continue
            
            # Get weights and compute projection
            W = clf.get_weights()
            Ws.append(W)
            P_rowspace_wi = get_rowspace_projection(W)
            rowspace_projections.append(P_rowspace_wi)
            
    # Compute final projection matrix
    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    
    return P, rowspace_projections, Ws












