from typing import NoReturn
# from ...base import BaseEstimator
from ...base import BaseEstimator

import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = len(y)

        # fit classes
        self.classes_ = np.unique(y)
        n_k = [np.sum(y == k) for k in self.classes_]  # |{i:y_i=k}| for each label k
        K = len(self.classes_)

        # fit mu
        self.fit_mu(X, K, n_k, y)

        # fit sigma matrix - unbiased
        self.fit_cov_matrix(K, X, m, y)

        # fit cov_inv matrix
        self._cov_inv = inv(self.cov_)

        # fit pi
        self.pi_ = np.zeros(K)
        for i in range(len(self.classes_)):
            self.pi_[i] = n_k[i] / m

        self.fitted_ = True

    def fit_cov_matrix(self, K, X, m, y):

        temp_cov = []
        for i in range(m):
            k_index = np.where(self.classes_ == y[i])
            mu_yi_MLE = self.mu_[k_index]
            vec = X[i] - mu_yi_MLE
            temp_cov.append(np.outer(vec, vec) / (m - K))
        self.cov_ = np.array(np.sum(temp_cov, axis=0))

    def fit_mu(self, X, k, n_k, y):

        temp_mu = []
        for i, label in enumerate(self.classes_):

            # select and sum the rows i in X when y[i] = current label
            X_relevant_rows = X[y == label]
            sum_relevant_x = np.sum(X_relevant_rows, axis=0)
            # calculate the MLE for the current label and add it to self.mu
            temp_mu.append(sum_relevant_x / n_k[i])
        self.mu_ = np.array(temp_mu)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # returns the argmax of each row of the likelihood matrix
        index = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[index]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m = len(X)
        K = len(self.classes_)

        likelihoods = []
        for k in range(K):
            a_k = self._cov_inv @ self.mu_[k]
            b_k = np.log(self.pi_[k]) - ((self.mu_[k].T @ self._cov_inv @ self.mu_[k])/2)
            likelihoods.append((a_k @ X.T) + b_k)

        return np.array(likelihoods).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))