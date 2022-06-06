from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    rand = np.random
    f = lambda x: ((x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2))
    X_samples = np.linspace(-1.2, 2, n_samples)
    y_noiseless = f(X_samples)
    epsilon = rand.normal(0, noise, n_samples)
    y_samples = y_noiseless + epsilon

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X_samples), pd.Series(y_samples), 2 / 3)
    train_X, train_y, test_X, test_y = np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)
    fig1 = go.Figure([go.Scatter(x=np.concatenate(train_X), y=train_y, name="train", mode="markers"),
                      go.Scatter(x=np.concatenate(test_X), y=test_y, name="test", mode="markers"),
                      go.Scatter(x=X_samples, y=[f(x) for x in X_samples], name="noiseless", mode="markers")])
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    average_training_errors = []
    average_valid_errors = []

    for i in range(11):
        current_poly = PolynomialFitting(i)
        train_err, valid_err = cross_validate(current_poly, np.array(train_X), np.array(train_y), mean_square_error)
        average_training_errors.append(train_err)
        average_valid_errors.append(valid_err)

    fig2 = go.Figure([go.Scatter(x=np.arange(11), y=average_training_errors, name="average training errors"),
                      go.Scatter(x=np.arange(11), y=average_valid_errors, name="average validation errors")])
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = int(np.argmin(average_valid_errors))
    poly = PolynomialFitting(k_star)
    poly.fit(X_samples, y_samples)  # todo with noise or not?
    test_err = poly.loss(np.concatenate(test_X), test_y)
    print(f"k value: {k_star}, test error: {test_err}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, test_X = np.split(X, [n_samples], axis=0)
    train_y, test_y = np.split(y, [n_samples], axis=0)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range = np.linspace(0.001, 2, n_evaluations)
    train_errors_ridge, valid_errors_ridge = [], []
    train_errors_lasso, valid_errors_lasso = [], []
    best_param_ridge, best_param_lasso = 0, 0
    best_err_ridge, best_err_lasso = np.inf, np.inf

    for i in lam_range:
        ridge = RidgeRegression(lam=i)
        lasso = Lasso(alpha=i)
        train_err_ri, valid_err_ri = cross_validate(ridge, train_X, train_y, mean_square_error)
        train_err_lass, valid_err_lass = cross_validate(lasso, train_X, train_y, mean_square_error)
        if valid_err_ri < best_err_ridge:
            best_err_ridge = valid_err_ri
            best_param_ridge = i
        if valid_err_lass < best_err_lasso:
            best_err_lasso = valid_err_lass
            best_param_lasso = i

        train_errors_ridge.append(train_err_ri)
        valid_errors_ridge.append(valid_err_ri)
        train_errors_lasso.append(train_err_lass)
        valid_errors_lasso.append(valid_err_lass)

    fig = go.Figure([go.Scatter(x=lam_range, y=train_errors_ridge, name="ridge - train"),
                     go.Scatter(x=lam_range, y=valid_errors_ridge, name="ridge - validation"),
                     go.Scatter(x=lam_range, y=train_errors_lasso, name="lasso - train"),
                     go.Scatter(x=lam_range, y=valid_errors_lasso, name="lasso - validation")])
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    print(f"Best parameter for ridge model: {best_param_ridge}")
    print(f"Best parameter for lasso model: {best_param_lasso}")
    best_ridge = RidgeRegression(lam=float(best_param_ridge))
    best_lasso = Lasso(alpha=best_param_lasso)
    lin = LinearRegression()
    best_ridge.fit(train_X, train_y)
    best_lasso.fit(train_X, train_y)
    lin.fit(train_X, train_y)
    print(f"Test errors of the Ridge model: {best_ridge.loss(test_X, test_y)}.")
    print(f"Test errors of the Lasso model: {mean_square_error(test_y, best_lasso.predict(test_X))}.")
    print(f"Test errors of the linear model: {lin.loss(test_X, test_y)}.")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
