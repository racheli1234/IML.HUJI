from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    df = pd.get_dummies(data=df, columns=['zipcode'], drop_first=True)
    # drop the columns of id and date
    df.drop(['id', 'date'], axis=1, inplace=True)
    # drop rows of houses whose price is 0
    df = df[df['price'] > 0]
    y = df.pop('price')
    return df, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    p_correlation = []
    for col in X.columns:

        correlation = y.cov(X[col]) / (np.sqrt(np.var(X[col]) * np.var(y))) #todo
        p_correlation.append(correlation)
        fig = go.Figure([go.Scatter(x=X[col], y=y, mode="markers")],
                        layout=dict(title=f"correlation between {col} and response = {correlation}"))
        fig.show()

    # fig = go.Figure([go.Scatter(x=X.columns, y=p_correlation)],
    #                 layout=dict(title="q2", xaxis_title="feature", yaxis_title="correlation between feature and "
    #                                                                            "response"))
    # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:\\Users\\Home\\Desktop\\studies\\IML\\IML.HUJI\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, .75)
    # print("training: \n", split_train_test(X, y)[0])
    # print("train y:\n", split_train_test(X, y)[1])
    # print("test:\n",split_train_test(X, y)[2])
    # print("test y:\n", split_train_test(X, y)[3])

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    for p in range(10, 101):
        samples = train_X.sample(frac=p/100, random_state=0)
        sample_y = train_y.sample(frac=p/100, random_state=0)


