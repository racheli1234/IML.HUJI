from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        min_loss = 1

        for feature in range(X.shape[1]):
            best_threshold_positive, min_loss_positive = self._find_threshold(X[:, feature], y, 1)
            best_threshold_negative, min_loss_negative = self._find_threshold(X[:, feature], y, -1)
            if min_loss_positive < min_loss_negative:
                min_feature_loss = min_loss_positive
                best_feature_threshold = best_threshold_positive
                cur_sign = 1
            else:
                min_feature_loss = min_loss_negative
                best_feature_threshold = best_threshold_negative
                cur_sign = -1

            if min_feature_loss <= min_loss:
                min_loss = min_feature_loss
                self.j_ = feature
                self.sign_ = cur_sign
                self.threshold_ = best_feature_threshold

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        relevant_feature_col = X[:, self.j_]
        m = X.shape[0]  # num of samples
        prediction = np.zeros(m)
        for i in range(m):
            if relevant_feature_col[i] < self.threshold_:
                prediction[i] = -1 * self.sign_
            else:
                prediction[i] = self.sign_

        return prediction

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_indexes = values.argsort()
        labels = labels[sorted_indexes]
        values = values[sorted_indexes]
        current_loss = self.weighted_misclassification(np.ones(len(labels)) * sign, labels)
        threshold_index = 0
        final_loss = current_loss

        for i in range(len(values)):
            # if the true label of the sample which we just put left to the threshold
            # (that is we labeled it as -sign) is sign, we add it to the loss:
            if 0 < i:
                if np.sign(labels[i - 1]) == sign:
                    current_loss += abs(labels[i - 1])
                else:  # it means that before we change the threshold its label was wrong, and now it's true so we
                    # decrease the loss
                    current_loss -= abs(labels[i - 1])
            if current_loss < final_loss:
                final_loss = current_loss
                threshold_index = i

        return values[threshold_index], final_loss / np.sum(abs(labels))  # todo

    def weighted_misclassification(self, y_pred: np.ndarray, y_true: np.ndarray):
        loss = 0
        for i in range(len(y_true)):
            if np.sign(y_true[i]) != np.sign(y_pred[i]):
                loss += abs(y_true[i])

        return loss

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
        return self.weighted_misclassification(self._predict(X), y)

if __name__ == '__main__':
    stump = DecisionStump()
    a = np.array(
        [[1, 1], [2, 2], [4, 0], [5, 1], [6, 3], [7, 4], [8, 5], [9, 6],
         [10, 7], [12, 8]])
    b = np.array([-1, 1, -1, -1, 1, 1, 1, 1, 1, 1])
    stump.fit(a, b)
    print(stump.j_, stump.sign_, stump.threshold_)