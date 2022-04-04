import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime, date

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    # add a column DayOfYear
    df['DayOfYear'] = df['Date'].dt.dayofyear
    # remove impossible temp
    df = df[df['Temp'] > -60]
    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("C:\\Users\\Home\\Desktop\\studies\\IML\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # select the rows of israel
    israel = X.loc[X['Country'] == 'Israel']
    israel["Year"] = israel["Year"].astype(str)  # TODO
    # 2(a)
    fig = px.scatter(israel, x="DayOfYear", y="Temp", title="temperature in Israel as a function of day of year",
                     color='Year')
    # fig.show()
    # 2(b)
    months = israel.groupby('Month').agg({"Temp": "std"})
    fig1 = px.bar(data_frame=months, y="Temp", barmode="group", title='STD of temperature in Israel as a function of '
                                                                      'month')
    fig1.update_yaxes(title_text="std")
    fig1.show()

    # Question 3 - Exploring differences between countries
    months_all = X.groupby(['Month', 'Country']).Temp.agg(["std", "mean"]).reset_index()
    fig3 = px.line(data_frame=months_all, y="mean", x=["Month"], color="Country", error_y="std",
                   title='the average and std monthly temperature of each country')  # todo x title
    # fig3.show()

    # Question 4 - Fitting model for different values of `k`
    israel_train_X, israel_train_y, israel_test_X, israel_test_y = split_train_test(israel["DayOfYear"],
                                                                                    israel["Temp"], 0.75)
    losses = []
    degs = np.linspace(1, 10, 10)
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(israel_train_X.values, israel_train_y.values)
        losses.append(round(poly.loss(israel_test_X.values, israel_test_y.values), 2))
    print(losses)
    fig4 = px.bar(x=degs, y=losses, title='Test error as a function of the value of k')
    fig4.update_xaxes(title_text='k')
    fig4.update_yaxes(title_text='The test error')
    # fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_deg_k_israel = PolynomialFitting(5)
    poly_deg_k_israel.fit(israel["DayOfYear"], israel["Temp"])
    countries_losses = []
    countries_list = X['Country'].unique().tolist()
    countries_list.remove('Israel')
    for country in countries_list:
        cur_country = X[X['Country'] == country]
        countries_losses.append(poly_deg_k_israel.loss(cur_country['DayOfYear'], cur_country['Temp']))
    fig5 = px.bar(x=countries_list, y=countries_losses, title='The k-deg polynomial model’s error over each of the '
                                                              'other countries')
    fig5.update_xaxes(title_text='country')
    fig5.update_yaxes(title_text='error')
    # fig5.show()
