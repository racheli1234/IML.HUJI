from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    uni = UnivariateGaussian()
    uni.fit(samples)
    print(uni.get_mu(), uni.get_var())

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int64)
    absolute_distance = []
    for m in ms:
        X = np.random.normal(10, 1, size=m)
        absolute_distance.append(abs(np.mean(X) - 10))
    fig2 = go.Figure([go.Scatter(x=ms, y=absolute_distance)],
                    layout=dict(title=r"$\text{absolute distance between the estimated- and true value of the "
                                           r"expectation, as a function of the sample size}$",
                                xaxis_title="m - number of samples",
                                yaxis_title="r$|\hat\mu - \mu|$"))
    fig2.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    # x_axis = np.linspace(1, 1000, 1000).astype(np.int64)
    pdfs = uni.pdf(samples)
    fig3 = px.scatter(x=samples, y=pdfs, title="empirical PDF function under the fitted model")
    fig3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu = [0, 0, 4, 0]
    cov_mat = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    multivar_samples = np.random.multivariate_normal(mu, cov_mat, 1000)
    multi = MultivariateGaussian()
    multi.fit(multivar_samples)
    print(multi.get_mu(), multi.get_cov())

    # Question 5 - Likelihood evaluation

    values = np.linspace(-10, 10, 200)
    log_likelihoods = np.zeros((200, 200))
    for i in range(len(values)):
        for j in range(len(values)):
            mu5 = [values[i], 0, values[j], 0]
            log_likelihoods[i][j] = MultivariateGaussian.log_likelihood(mu5, cov_mat, multivar_samples)
    go.Figure(data=go.Heatmap(x=values,y=values,z=log_likelihoods, colorbar=dict(title="log likelihood value")),
              layout=dict(title="Question 5: log likelihood of models with mu = [f1,0,f3,0] and the covariance from Q4",
                          xaxis_title="f3",
                          yaxis_title="f1")).show()

    # Question 6 - Maximum likelihood



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
