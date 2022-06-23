import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.model_selection

import IMLearn
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from IMLearn.metrics.loss_functions import misclassification_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    vals = []
    weights = []

    def callback(cur_w, cur_val):
        vals.append(cur_val)
        weights.append(cur_w)

    return callback, vals, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module in [L1, L2]:
        for eta in etas:
            f = module(init.copy())
            callback, values, weights = get_gd_state_recorder_callback()
            lr = FixedLR(eta)
            gd = GradientDescent(learning_rate=lr, callback=callback, out_type="last")
            gd.fit(f=f, X=None, y=None)
            print(f"lowest loss of {module.__name__} with eta = {eta}: {np.min(values)}")
            plot_descent_path(module, np.array(weights), f"with eta = {eta}, module = {module.__name__}").show()
            fig = go.Figure(data=[go.Scatter(x=np.arange(len(values)), y=values, mode="markers")])
            fig.update_layout(title=f"Convergence rate of GD with module = {module.__name__}, eta = {eta}")
            fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    all_vals = []
    for gamma in gammas:
        f = L1(init.copy())
        callback, values, weights = get_gd_state_recorder_callback()
        lr = ExponentialLR(base_lr=eta, decay_rate=gamma)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(f=f, X=None, y=None)
        all_vals.append(values)
        if gamma == .95:
            plot_descent_path(L1, np.array(weights), f"with gamma = 0.95, module = L1").show()

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure(data=[go.Scatter(x=np.arange(len(all_vals[i])), y=all_vals[i], name=f"gamma={gammas[i]}")
                          for i in range(len(gammas))])
    fig.update_layout(title="Convergence rate of L1 for all decay rates")
    fig.show()
    print([f"min loss of L1 with gamma = {gammas[i]}: {np.min(all_vals[i])}\n" for i in range(len(gammas))])

    # Plot descent path for gamma=0.95
    # in the for loop


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    logistic = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    logistic.fit(np.array(X_train), np.array(y_train))
    fpr, tpr, thresholds = roc_curve(y_train, logistic.predict_proba(np.array(X_train)))
    fig = go.Figure(
        data=[go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    best_alpha = np.round(thresholds[np.argmax(tpr - fpr)], 2)
    print("best alpha: ", best_alpha)
    # print("Test error with the best alpha: ")

    # Plotting convergence rate of logistic regression over SA heart disease data

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from IMLearn.model_selection import cross_validate
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for L in ["l1", "l2"]:
        valid_score_list = []
        train_score_list = []
        for lam in lambdas:
            log = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                     penalty=L, lam=lam)
            train_score, valid_score = cross_validate(log, np.array(X_train), np.array(y_train),
                                                      scoring=misclassification_error)
            valid_score_list.append(valid_score)
            train_score_list.append(train_score)
        best_lam_index = np.argmin(valid_score_list)
        best_lam = lambdas[best_lam_index]
        logistic_with_best_lam = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                               penalty=L, lam=best_lam)
        logistic_with_best_lam.fit(np.array(X_train), np.array(y_train))
        test_err = logistic_with_best_lam.loss(np.array(X_test), np.array(y_test))
        print(f"{L} model: Best lambda is {best_lam}. Test error is {test_err}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
