""" CSC110 Fall 2021 Final Project: Simple Linear Regression

Module Description
==================
This Python file implements a simple linear regression algorithm in order to investigate the
relationship between the number of COVID-19 hospitalizations and the death counts for three major
diseases in Ontario, the values of which are given by an imported dataset. This file will calculate
the regression line which best describes the data.

Copyright and Usage Information
===============================
This file is provided solely for the use of grading the final project of CSC110 by
the TA's and instructors of the department of Computer Science at the University of
Toronto St. George campus. Modification, usage and distribution of this code for any
other purpose is prohibited.

This file is Copyright (c) 2021 Anna Lee Pantoja, Savanna Pan, Tanvi Patel, Vidhi Patel.

"""
from load_data import group_datasets


def retrieve_data() -> tuple:
    """Return a tuple containing the values for number of hospitalizations and the death counts for
    three major diseases, combined.
    """
    data = group_datasets()

    # Collect hospitalization data and total death counts
    hosp = data['COVID-19 Hospitalizations'].values
    sum_death_counts = data['Total d.c.'].values

    return hosp, sum_death_counts


def generate_best_fit_line() -> tuple:
    """Implements a simple linear regression algorithm to return a tuple containing the slope and
    y-intercept of the regression line for the data.

    The equation of a line is given by y = mx + b, where m is the slope and b is the y-intercept.

    The regression line is calculated using the method of least squares, where the slope, m, and
    the y-intercept, b, are calculated using the following formulas:

    m = sum((x_0 - mean(x)) * (y_0 - mean(y))) / sum((x_0 - mean(x))^2)

    b = mean(y) - (m * mean(x))

        where x_0 is the values of the x-variable, y_0 is the values of the y-variable, mean(x) is
        the mean of all x-values, and mean(y) is the mean of all y_values.

    The hospitalization values will be considered as the x-values and the death counts as the
    y-values.
    """
    # x and y values
    hosp, death_counts = retrieve_data()

    # Calculate the mean of the x-values and the y-values
    mean_hosp = sum(hosp) / len(hosp)
    mean_death_counts = sum(death_counts) / len(death_counts)

    total_values = len(hosp)

    # Calculate the slope of the regression line
    # ACCUMULATORS: to keep track of values added to the numerator and denominator
    num = 0
    den = 0

    for i in range(0, total_values):
        num = num + ((hosp[i] - mean_hosp) * (death_counts[i] - mean_death_counts))
        den = den + (hosp[i] - mean_hosp) ** 2

    slope = num / den

    # Calculate the y-intercept of the regression line
    y_int = mean_death_counts - (slope * mean_hosp)

    coeff = (slope, y_int)

    return coeff


# if __name__ == '__main__':
#     import python_ta
#     import python_ta.contracts
#
#     python_ta.contracts.DEBUG_CONTRACTS = False
#     python_ta.contracts.check_all_contracts()
#
#     python_ta.check_all(config={
#         'allowed-io': [],
#         'extra-imports': ['python_ta.contracts', 'load_data'],
#         'max-line-length': 100,
#         'disable': ['R1705', 'C0200'],
#     }, output='pyta_report.html')
