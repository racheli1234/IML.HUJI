from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    m = len(X)
    # create a list of indices that separate between S_1,S_2,...,S_cv
    split_indexes = [(m // cv) * i for i in range(cv)]
    split_indexes.append(m)

    train_scores_sum = 0
    validation_scores_sum = 0
    for i in range(cv):

        s_i_range = np.arange(split_indexes[i], split_indexes[i + 1])
        train_X, train_y = np.delete(X, s_i_range, 0), np.delete(y, s_i_range, 0)
        validation_X, validation_y = X[s_i_range], y[s_i_range]
        estimator.fit(train_X, train_y)
        train_scores_sum += scoring(train_y, estimator.predict(train_X))
        validation_scores_sum += scoring(validation_y, estimator.predict(validation_X))

    return train_scores_sum / cv, validation_scores_sum / cv


