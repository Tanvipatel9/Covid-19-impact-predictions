""" CSC110 Fall 2021 Final Project: Visualization

Module Description
==================
This Python file creates two plots that graphically represent the relationship between variables
in the predetermined dataset. Each can be displayed by calling the corresponding function.

One is a scatter plot with the weekly number of COVID-19 hospitalizations on the x-axis and the
combined weekly death counts of three major diseases on the y-axis. This scatter plot also includes
a regression line generated for this data.

The second plot is a heatmap that displays the correlation coefficient calculated for each pair of
variables in the dataset, where the values of interest are those that describe the correlation
between each of the death counts with the number of COVID-19 hospitalizations. A helper function is
included in this file, whose purpose is to generate an array containing the correlation coefficients
for this plot.

Copyright and Usage Information
===============================
This file is provided solely for the use of grading the final project of CSC110 by
the TA's and instructors of the department of Computer Science at the University of Toronto
St. George campus. Modification, usage and distribution of this code for any other purpose
is prohibited.

This file is Copyright (c) 2021 Anna Lee Pantoja, Savanna Pan, Tanvi Patel, Vidhi Patel.
"""
import matplotlib.pyplot as plt
import numpy as np
from load_data import get_dataset_values, group_datasets
from correlation import pearson
from linear_reg import retrieve_data, generate_best_fit_line

DATA = get_dataset_values()


###################################################################################################
# Visualization for regression line (Scatter plot + regression line)
###################################################################################################


def plot_regression_line() -> None:
    """Creates and displays a scatter plot with a regression line that attempts to describe a linear
     relationship between the data.

     Here, the x-values are the number of COVID-19 hospitalizations and are represented as hosp,
     and the y-values are the combined death counts of the three major diseases in the final
     dataset and are represented as death_counts.

     A line is given by the formula y = mx + b, where m is the slope and b is the y-intercept.
     These values are obtained from another module.
    """
    hosp, death_counts = retrieve_data()

    # Retrieve the regression line's m and b
    slope, y_int = generate_best_fit_line()

    # Label the x and y axis and add a title
    plt.xlabel('Hospitalizations')
    plt.ylabel('Total Death Counts')
    plt.title('COVID-19 Hospitalizations vs Major Disease Deaths')

    # Create a scatter plot for the given data points with green square markers
    plt.scatter(hosp, death_counts, color='g', marker="s", s=22)

    # Predicted death counts using y = mx + b
    predicted_death = (slope * hosp) + y_int

    # Plot the regression line as a blue solid line
    plt.plot(hosp, predicted_death, color='b', linewidth=2)

    # Display the scatter plot and regression line
    plt.show()


###################################################################################################
# Visualization for correlations (Heatmap)
###################################################################################################


def plot_correlations() -> None:
    """Creates and displays a heatmap to visually represent the strength of the relationship
    between each of the variables in the dataset. The correlation coefficient for each pair of
    variables is also included in this heatmap.
    """
    correlations = get_correlations()  # The correlation coefficients for each pair of variables

    # Generate the labels for the plot
    labels = list(group_datasets().columns)
    row = labels
    column = labels

    # Input the data into the heatmap
    plt.imshow(correlations, cmap='Greens', interpolation='none')

    # Set the horizontal and vertical labels
    plt.xticks([0, 1, 2, 3, 4], labels=column, rotation='vertical')
    plt.yticks([0, 1, 2, 3, 4], labels=row)

    # Add the color bar and title
    plt.colorbar()
    plt.title('Correlation Between Variables')

    # Add the correlation values to the heatmap
    for y in range(len(column)):
        for x in range(len(row)):
            plt.text(x, y, '{:.3f}'.format(correlations[y, x]), ha='center', va='center',
                     size='x-small')

    # Display the correlation plot
    plt.show()


def get_correlations() -> np.array:
    """Return an array of a list of lists, where each inner list contains the values of the
     correlation coefficients for the relationship between each variable and other variables
     in the imported dataset.
     """
    # ACCUMULATOR outer_lst: keeps track of the inner lists
    outer_lst = []

    for variable_v1 in DATA:
        # ACCUMULATOR inner_lst: keeps track of the correlation
        # coefficients between a variable and the others
        inner_lst = []

        for variable_v2 in DATA:
            value = pearson(variable_v1, variable_v2)
            inner_lst.append(value)

        outer_lst.append(inner_lst.copy())

    return np.array(outer_lst)


# if __name__ == '__main__':
#     import python_ta
#     import python_ta.contracts
#
#     python_ta.contracts.DEBUG_CONTRACTS = False
#     python_ta.contracts.check_all_contracts()
#
#     python_ta.check_all(config={
#         'allowed-io': [],
#         'extra-imports': ['python_ta.contracts', 'load_data', 'correlation', 'linear_reg',
#                           'matplotlib.pyplot', 'numpy'],
#         'max-line-length': 100,
#         'disable': ['R1705', 'C0200'],
#     }, output='pyta_report.html')
