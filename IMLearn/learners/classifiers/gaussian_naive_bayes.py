from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        # fit cov
        self.fit_var_matrix(X, n_k, y)

        # fit pi
        self.pi_ = np.zeros(K)
        for i in range(len(self.classes_)):
            self.pi_[i] = n_k[i] / m

        self.fitted_ = True

    def fit_var_matrix(self, X, n_k, y):
        vars = []
        for index, k in enumerate(self.classes_):
            X_relevant_rows = X[y == k]
            kth_row_content = []
            for i in range(len(X_relevant_rows)):
                kth_row_content.append(np.square(X_relevant_rows[i] - self.mu_[k]))
            vars.append(np.sum(np.array(kth_row_content), axis=0) / (n_k[index] - 1))
        self.vars_ = np.array(vars)

    def fit_mu(self, X, k, n_k, y):
        self.mu_ = np.zeros((k, X.shape[1]))
        for i, label in enumerate(self.classes_):
            # select and sum the rows i in X when y[i] = current label
            X_relevant_rows = X[y == label]
            sum_relevant_x = np.sum(X_relevant_rows, axis=0)
            # calculate the MLE for the current label and add it to self.mu
            self.mu_[i] = sum_relevant_x / n_k[i]

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
        m = X.shape[0]
        d = X.shape[1]
        K = len(self.classes_)

        likelihoods = []
        for k in range(K):
            part_1 = np.log(self.pi_[k])
            part_2_list = []
            for j in range(d):
                sigma_k_j = self.vars_[k][j]
                mu_k_j = self.mu_[k][j]
                part_2_list.append(
                    (np.log(np.sqrt(2 * np.pi * sigma_k_j))) + ((np.square((X[:, j] - mu_k_j)) / sigma_k_j) / 2))
            part_2 = np.sum(np.array(part_2_list), axis=0)
            likelihoods.append(part_1 - part_2)

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
