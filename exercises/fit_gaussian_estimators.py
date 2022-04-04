from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

pio.templates.default = "simple_white"
Q2_TITLE = "absolute distance between the estimated- and true value of the " \
           "expectation, as a function of the sample size"
Q2_X_TITLE = "m - number of samples"
Q2_Y_TITLE = "r$|\hat\mu - \mu|$"
Q3_TITLE = "empirical PDF function under the fitted model"
Q5_TITLE = "Question 5: log likelihood of models with mu = [f1,0,f3,0] and the covariance from Q4"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    samples = np.random.normal(mu, var, 1000)
    uni = UnivariateGaussian()
    uni.fit(samples)
    print(uni.get_mu(), uni.get_var())

    # Question 2 - Empirically showing sample mean is consistent
    x_arr = np.linspace(10, 1000, 100).astype(np.int64)
    absolute_distance = []

    X = UnivariateGaussian()
    for i in range(1, 100):
        X.fit(samples[:i * 10])
        absolute_distance.append(abs(X.get_mu() - mu))
    fig2 = go.Figure([go.Scatter(x=x_arr, y=absolute_distance)],
                     layout=dict(title=Q2_TITLE,
                                 xaxis_title=Q2_X_TITLE,
                                 yaxis_title=Q2_Y_TITLE))
    fig2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni.pdf(samples)
    data = pd.DataFrame(dict(samples=samples, pdfs=pdfs))
    fig3 = px.scatter(data, x="samples", y="pdfs", title=Q3_TITLE)
    fig3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu4 = [0, 0, 4, 0]
    cov_mat = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    multivar_samples = np.random.multivariate_normal(mu4, cov_mat, 1000)
    multi = MultivariateGaussian()
    multi.fit(multivar_samples)
    print(multi.get_mu())
    print(multi.get_cov())

    # Question 5 - Likelihood evaluation

    values = np.linspace(-10, 10, 200)
    log_likelihoods = np.zeros((200, 200))
    argmax = (0, 0)
    max_val = np.NINF
    for i in range(len(values)):
        for j in range(len(values)):
            mu5 = [values[i], 0, values[j], 0]
            log_likelihood_val = MultivariateGaussian.log_likelihood(mu5, cov_mat, multivar_samples)
            log_likelihoods[i][j] = log_likelihood_val
            if log_likelihood_val > max_val:
                max_val = log_likelihood_val
                argmax = (values[i], values[j])

    go.Figure(data=go.Heatmap(x=values, y=values, z=log_likelihoods, colorbar=dict(title="log likelihood value")),
              layout=dict(title=Q5_TITLE,
                          xaxis_title="f3",
                          yaxis_title="f1")).show()

    # Question 6 - Maximum likelihood
    print("max value: ", max_val, "\nargmax: ", argmax)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
