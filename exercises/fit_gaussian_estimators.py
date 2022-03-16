from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10,1,1000)
    uni = UnivariateGaussian()
    uni.fit(samples)
    print(uni.get_mu(), uni.get_var())

    # Question 2 - Empirically showing sample mean is consistent
    # ms = np.linspace(10, 1000, 100).astype(np.int64)
    # absolute_distance = []
    # for m in ms:
    #     X = np.random.normal(10, 1, size=m)
    #     absolute_distance.append(abs(np.mean(X)-10))
    # fig = go.Figure([go.Scatter(x=ms, y=absolute_distance)],
    #           layout=go.Layout(title=r"$\text{absolute distance between the estimated- and true value of the "
    #                                  r"expectation, as a function of the sample size}$"))
    #todo add axis titles

    # fig.show()
              # xaxis_title="$m\\text{ - number of samples}$",
              # yaxis_title="r$|\hat\mu - mu|$",
              # height=300).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    x_axis = np.linspace(1, 1000, 1000).astype(np.int64)
    pdfs = uni.pdf(samples)
    fig1 = go.Figure([go.Scatter(x=x_axis, y=pdfs)], layout=go.Layout(title="jnfvjfnrvik"))
    fig1.show()


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
