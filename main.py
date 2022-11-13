""" CSC110 Fall 2021 Final Project: Run Program

Module Description
==================
When run, this Python file gives the user the option to select a data visualization to display.
The number input determines which plot (if any) will be displayed. To select a different option,
the file must be run again.

Copyright and Usage Information
===============================
This file is provided solely for the use of grading the final project of CSC110 by
the TA's and instructors of the department of Computer Science at the University of
Toronto St. George campus. Modification, usage and distribution of this code for any
other purpose is prohibited.

This file is Copyright (c) 2021 Anna Lee Pantoja, Savanna Pan, Tanvi Patel, Vidhi Patel.

"""
import visualization as vis

if __name__ == '__main__':

    # Print the Introduction

    print('We have attempted to find a relationship between the number of weekly hospitalizations \n'
          'due to COVID-19 and the number of weekly deaths from other kinds of diseases. \n')

    invalid_input = True

    print('Enter \"1\" to view the correlation table.')
    print('Enter \"2\" to view the regression line.')
    print('Rerun the program to reselect.\n')

    number = input()

    if number == '1':
        vis.plot_correlations()

    elif number == '2':
        vis.plot_regression_line()

    else:
        print('Invalid number, please enter a valid input\n')
