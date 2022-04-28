from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback_func(p: Perceptron, cur_x: np.ndarray, cur_y: int):
            losses.append(p._loss(X, y))

        percep = Perceptron(callback=loss_callback_func)
        percep._fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure([go.Scatter(x=np.arange(1, len(losses) + 1), y=np.array(losses))],
                        layout=dict(title=f"Loss value as a function of the iteration over {n} Data",
                                    xaxis_title="iteration", yaxis_title="loss"))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda._fit(X, y)
        lda_prediction = lda._predict(X)

        gnb = GaussianNaiveBayes()
        gnb._fit(X, y)
        gnb_prediction = gnb._predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_acc = accuracy(y, lda_prediction)
        gnb_acc = accuracy(y, gnb_prediction)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"model: Gaussian Naive Bayes estimator, accuracy: {gnb_acc}",
                                            f"model: LDA estimator, accuracy: {lda_acc}"))
        fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Models - {f} Dataset}}$")

        # Add traces for data-points setting symbols and colors
        models = [gnb, lda]
        predictions = [gnb_prediction, lda_prediction]
        symbols = np.array(["triangle-left", "circle", "square"])

        for i, p in enumerate(predictions):
            fig.add_trace(row=1, col=i + 1,
                          trace=go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                           marker=dict(color=p, symbol=symbols[y], line=dict(color="black", width=1)),
                                           showlegend=False))

        # Add `X` dots specifying fitted Gaussians' means
        for i, m in enumerate(models):
            fig.add_trace(row=1, col=i + 1,
                          trace=go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers",
                                           marker=dict(color="black", symbol="x", line=dict(color="black", width=1)),
                                           showlegend=False))

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            fig.add_trace(row=1, col=1,
                          trace=get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])))

            fig.add_trace(row=1, col=2,
                          trace=get_ellipse(lda.mu_[i], lda.cov_))
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)

    run_perceptron()
    compare_gaussian_classifiers()

    # import numpy as np
    # q1 quiz:
    # S = {(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)}
    # X = np.array([0,1,2,3,4,5,6,7])
    # y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    # g = GaussianNaiveBayes()
    # g._fit(X,y)
    # print("mu: ", g.mu_)
    # print("pi: ", g.pi_)

    # q2 quiz:

    # X2 = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    # y2 = np.array([0,0,1,1,1,1])
    # g2 = GaussianNaiveBayes()
    # g2._fit(X2, y2)
    # print("mu: ", g2.mu_)
    # print("pi: ", g2.pi_)
    # print("vars: ", g2.vars_)


