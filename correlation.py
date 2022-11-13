""" CSC110 Fall 2021 Final Project: Calculate the Pearson Correlation Coefficient

Module Description
==================
This Python file calculates the Pearson correlation coefficient, a linear correlation coefficient
which quantifies the strength and direction of the relationship between two variables.
This coefficient can take a value between 1.0 and -1.0, where values approaching 1.0 indicate a
strong positive correlation and values approaching -1.0 indicate a strong negative correlation.
Values approaching 0 indicates no linear relationship.

Copyright and Usage Information
===============================
This file is provided solely for the use of grading the final project of CSC110 by
the TA's and instructors of the department of Computer Science at the University of
Toronto St. George campus. Modification, usage and distribution of this code for any
other purpose is prohibited.

This file is Copyright (c) 2021 Anna Lee Pantoja, Savanna Pan, Tanvi Patel, Vidhi Patel.

"""
import math


def pearson(x_values: list, y_values: list) -> float:
    """Calculates the Pearson correlation coefficient (Pearson's r)

    This value is calculated using the following formula:

    r = (n * sum(x_0 * y_0) - (sum(x_0) * sum(y_0))) /
        sqrt((n * sum(x_0^2) - sum(x_0)^2) * (n * sum(y_0^2) - sum(y_0)^2))

        where r is the coefficient, x_0 is the values of the x-variable, and y_0 is
        the values of the y-variable.

    Preconditions:
      - len(x_values) == len(y_values)
    """
    n = len(x_values)

    # ACCUMULATORS: to keep track of the sums of (x*y), x^2, and y^2
    xy_so_far = 0
    x_squared_so_far = 0
    y_squared_so_far = 0

    for i in range(n):
        xy_so_far += x_values[i] * y_values[i]
        x_squared_so_far += x_values[i] ** 2
        y_squared_so_far += y_values[i] ** 2

    # The numerator of the formula
    num = (n * xy_so_far) - ((sum(x_values)) * (sum(y_values)))

    # The denominator of the formula
    den1 = (n * x_squared_so_far) - (sum(x_values)) ** 2
    den2 = (n * y_squared_so_far) - (sum(y_values)) ** 2
    den = math.sqrt(den1) * math.sqrt(den2)

    # return r
    return num / den


# if __name__ == '__main__':
#     import python_ta
#     import python_ta.contracts
#
#     python_ta.contracts.DEBUG_CONTRACTS = False
#     python_ta.contracts.check_all_contracts()
#
#     python_ta.check_all(config={
#         'allowed-io': [],
#         'extra-imports': ['python_ta.contracts', 'math'],
#         'max-line-length': 100,
#         'disable': ['R1705', 'C0200'],
#     }, output='pyta_report.html')
