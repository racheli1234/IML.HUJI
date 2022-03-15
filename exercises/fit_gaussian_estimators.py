from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10,1,1000)
    uni = UnivariateGaussian(False)
    uni.fit(samples)
    print(uni.get_mu(), uni.get_var())
    # raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 100, 10).astype(np.int64)
    absolute_distance = []
    for m in ms:
        X = np.random.normal(10, 1, size=m)
        absolute_distance.append(abs(np.mean(X)-10))
    go.Figure([go.Scatter(x=ms, y=absolute_distance)],
              layout=go.Layout(title=r"$\text{absolute distance between the estimated- \
                                        and true value of the expectation,\
                                        as a function of the sample size}$")).show()
              # xaxis_title="$m\\text{ - number of samples}$",
              # yaxis_title="r$|\hat\mu - mu|$",
              # height=300).show()

    # raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
