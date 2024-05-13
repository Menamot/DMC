

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin

class DMC(BaseEstimator, ClassifierMixin):
    def __int__(
            self,
            discretization='kmeans'
    ):
        self.discretization = discretization

    def fit(self, X, y, L=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().ravel()  # Use ravel() to make sure that y is one-dimensional

        K = len(np.unique(y))
        if L is None:
            L = np.ones((K,K)) - np.eye(K)

        if self.discretization == 'kmeans':
            # self.T_optimal =
            self.discretization_model = KMeans(n_clusters=self.T_optimal)
            self.discretization_model.fit(X)
            self.profile_labels = self.discretization_model.labels_

        if self.discretization == "DT":
            pass

        self.pHat = compute_pHat(self.profile_labels, y, K, self.T_optimal)
        # self.piStar = compute_piStar()

    def predict(self, X):
        pass

    def predict_prob(self, X):
        pass


def compute_pi(y:np.ndarray, K:int):
    """

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    Returns
    -------
    pi : ndarray of shape (K,)
        Proportion of classes
    """
    pi = np.zeros(K)
    total_count = len(y)

    for k in range(K):
        pi[k] = np.sum(y == k) / total_count
    return pi

def compute_pHat(profile_labels:np.ndarray, y:np.ndarray, K:int, T:int):
    """

    Parameters
    ----------
    profile_labels : ndarray of shape (n_samples,)
        Labels of profiles for each data point

    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    T : int
        Number of profiles

    Returns
    -------
    pHat : ndarray of shape(K, n_profiles)
    """
    pHat = np.zeros((K, T))

    for k in range(K):
        Ik = np.where(y == k)[0]
        mk = len(Ik)
        for t in range(T):
            pHat[k, t] = np.sum(profile_labels[Ik] == t) / mk
    return pHat



def compute_conditional_risk(y_true:np.ndarray, y_pred:np.ndarray, K:int, L:np.ndarray):
    """

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Real labels

    y_pred : ndarray of shape (n_samples,)
        Predicted labels

    K : int
        Number of classes

    L : ndarray of shape (K, K)
        Loss matrix

    Returns
    -------
    R : ndarray of shape (K,)
        Conditional risk

    confmat : ndarray of shape (K,K)
        Confusion matrix
    """
    confmat = np.zeros((K, K))

    for k in range(K):
        Ik = np.where(y_true == k)[0]
        mk = len(Ik)
        if mk > 0:
            # Calculate the proportion of each predicted category in Ik
            for l in range(K):
                confmat[k, l] = np.sum(y_pred[Ik] == l) / mk
    # Calculate the conditional risk for each true category
    R = np.dot(L, confmat.T).diagonal().ravel()

    return R, confmat


