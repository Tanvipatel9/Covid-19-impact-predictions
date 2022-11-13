""" CSC110 Fall 2021 Final Project: Load, Clean, and Process Datasets

Module Description
==================
This Python file contains code to load two datasets that will be cleaned, transformed, and grouped
into one final dataset to prepare the data for use in other modules.

Copyright and Usage Information
===============================
This file is provided solely for the use of grading the final project of CSC110 by
the TA's and instructors of the department of Computer Science at the University of
Toronto St. George campus. Modification, usage and distribution of this code for any
other purpose is prohibited.

This file is Copyright (c) 2021 Anna Lee Pantoja, Savanna Pan, Tanvi Patel, Vidhi Patel.

"""
import csv
import datetime
import pandas as pd

# Set constants string/ dictionary to be used for the functions below

PUNCTUATION = '!\"#$%&()*+,-./:;<=>?@[]^_`{|}~'

MONTH_TO_INT = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
                'June': 6, 'July': 7, 'August': 8, 'September': 9,
                'October': 10, 'November': 11, 'December': 12}


###################################################################################################
# The final dataset
###################################################################################################
# This function will return the DataFrame containing the data that will be used in this project
#
# The observations are grouped weekly, with each timestamp in the DataFrame corresponding to the
# Saturday of that week (we consider a week to start on Sunday and end on Saturday).
# The index of the DataFrame consists of timestamps, ranging from the dates 2020-04-11 to
# 2021-07-31.
# The variables included are the number of deaths from malignant neoplasms, the number of deaths
# from diseases of the heart, the number of deaths from chronic lower respiratory disease, the
# number of deaths of those three diseases combined, and the number of hospitalizations of
# COVID-19 patients.


def group_datasets() -> any:
    """ Return a DataFrame containing the data from the processed datasets of weekly
    COVID-19 hospitalizations and weekly death counts from three major diseases combined.
    The columns in both datasets are extracted and then combined into a single DataFrame.
    """
    # Get the values from the columns of both datasets
    dataset_columns = get_dataset_values()

    # Group the values into a dict to input into the DataFrame
    grouped_dataset = {'Total d.c.': dataset_columns[0],
                       'Malignant Neoplasms d.c.': dataset_columns[1],
                       'Diseases of the Heart d.c.': dataset_columns[2],
                       'Chronic Lower Respiratory Diseases d.c.': dataset_columns[3],
                       'COVID-19 Hospitalizations': dataset_columns[4]}

    # Get the index (both datasets have the same dates as in their index)
    dataset_index = read_death_count_data()[0]

    # Group the columns from both datasets
    final_dataset = pd.DataFrame(grouped_dataset, index=dataset_index)

    return final_dataset  # the DataFrame object that can be used by other functions


###################################################################################################
# Loading and processing the two original datasets
###################################################################################################
# The two datasets under the filenames 'dataset1.csv' and 'dataset2.csv' are to be cleaned,
# transformed, and grouped before their data can be used by other modules.
#
# The first dataset, 'dataset1', contains provisional weekly death counts, grouped by province and
# by cause of death. The province that will be taken into account is Ontario and the causes of death
# we that will be included are malignant neoplasms, diseases of the heart, chronic lower respiratory
# disease. A column will be added with data on the total death counts of the three together.
# This data is processed using Python and the csv module.
#
# The second dataset, 'dataset2', contains daily data on hospitalizations of patients with
# COVID-19 in Ontario, grouped by region of Ontario and separated into different categories of
# hospitalizations. The regional data will be grouped to consider the entire province and the data
# will be aggregated from daily to weekly. The kind of hospitalization that will be considered is
# general hospitalizations.
# This data is processed using Python and the pandas library.


# Dataset 1
def read_death_count_data() -> list[list]:
    """Return a list containing lists based on the desired data from the file 'dataset1.csv'.

    The returned lists will include a list of the weekly timestamps, a list of the total weekly
    death counts from malignant neoplasms, diseases of the heart, and chronic lower respiratory
    disease combined, as well as three lists containing the weekly death counts of the individual
    diseases, all from from April 5th, 2020 to July 31st, 2021.
    """

    # Create all the lists needed for the computations below

    collect_all_rows = []
    mn_lst = []  # mn stands for 'malignant neoplasms'
    doh_lst = []  # doh stands for 'diseases of heart'
    clrd_lst = []  # clrd stands for 'chronic lower respiratory diseases'
    sum_lst = []  # sum of the death counts of the three diseases
    num_to_remove = []

    # These are individual lists to collect the death counts of each diseases including
    # malignant neoplasms, diseases of the heart and chronic lower respiratory diseases

    mn_total = []
    doh_total = []
    clrd_total = []

    # Open and read the csv file

    with open('dataset1.csv', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=',')

        for row in reader:
            collect_all_rows.append(row)  # Collect all rows from the csv file into a list

        dt_lst = create_datetime(collect_all_rows)  # dt stands for datetime

        # Collect data for only the three diseases that will be grouped

        for row in collect_all_rows:

            if 'Malignant neoplasms' in row:
                mn_lst.append(row)

            elif 'Diseases of heart' in row:
                doh_lst.append(row)

            elif 'Chronic lower respiratory diseases' in row:
                clrd_lst.append(row)

        # Remove the first 2 elements in the list, which are not death counts

        for _ in range(2):
            mn_lst[5].pop(0)
            doh_lst[5].pop(0)
            clrd_lst[5].pop(0)

        # Add the total number of death counts of the three diseases

        for i in range(len(mn_lst[5])):
            sum_lst.append(int(mn_lst[5][i]) + int(doh_lst[5][i]) + int(clrd_lst[5][i]))
            mn_total.append(int(mn_lst[5][i]))
            doh_total.append(int(doh_lst[5][i]))
            clrd_total.append(int(clrd_lst[5][i]))

        # Remove the extra dates that are out of the desired date range
        return_dates = remove_extra_dates(dt_lst)

        # Calculate how many extra dates need to be removed
        remove = len(sum_lst) - len(return_dates)

        mn_remove = []
        doh_remove = []
        clrd_remove = []

        # Append the numbers we need to remove into a remove list

        for i in range(remove):
            num_to_remove.append(sum_lst[i])
            mn_remove.append(mn_total[i])
            doh_remove.append(doh_total[i])
            clrd_remove.append(clrd_total[i])

        # Call the helper function to remove the extra numbers
        return_sum = remove_extra_nums(sum_lst, num_to_remove)
        return_mn = remove_extra_nums(mn_total, mn_remove)
        return_doh = remove_extra_nums(doh_total, doh_remove)
        return_clrd = remove_extra_nums(clrd_total, clrd_remove)

    # Return the final version of the list with the correct datetime and death counts
    # within our desired date range

    return [return_dates, return_sum, return_mn, return_doh, return_clrd]


# Dataset 2
def read_hospitalization_data() -> any:
    """Return a DataFrame containing the desired data from the file 'dataset2.csv'.

    The variables are weekly hospitalizations due to COVID-19, counted from April 5th, 2020 to
    July 31st, 2021.
    """

    # Read csv file for second dataset
    data2 = pd.read_csv('dataset2.csv')
    data2['date'] = pd.to_datetime(data2['date'])  # Convert 'date' from object to datetime64
    data2['date'].dt.normalize()  # Time component not relevant, so normalize

    # Pivot the DataFrame and include relevant information
    variables = ['date', 'oh_region', 'hospitalizations']  # The columns to include
    group_var = variables[:2]
    outcome_var = variables[2]
    data2 = data2.groupby(group_var, as_index=False)[outcome_var].sum()
    data2 = data2.set_index(['date', 'oh_region']).unstack('oh_region').fillna(0)  # Dates as index
    data2.columns = data2.columns.levels[1].rename(None)

    # Checking for complete days before aggregating as weekly data

    # (Uncommented) compare unique days in the dataset to the number of days in the date range
    # _days = len(data2.index.unique())  # Is 618
    date_range = pd.date_range(data2.index.min(), data2.index.max())
    # Data contains 618 unique days but the date range has 619, so 1 day is missing
    data2_new = data2.reindex(date_range, fill_value=0)  # Fill missing day's data with 0

    # Aggregate the number of hospitalizations to include observations for all of Ontario
    new_index = data2_new.index
    hosp_all_regions = sum(data2_new[column] for column in data2_new.columns)  # Concatenate
    data_hosp_on = pd.DataFrame({'Hospitalizations': hosp_all_regions}, index=new_index)

    # Isolate rows corresponding to the date range compatible with the death count dataset
    data_hosp_on = data_hosp_on['2020-04-05': '2021-07-31']

    data_hosp_w = data_hosp_on.resample('W-SAT').sum()  # Aggregate to weekly data (Sun-Sat)

    return data_hosp_w  # The processed dataset


###################################################################################################
# Helper functions to load the datasets
###################################################################################################


def create_datetime(data_input: list) -> list:
    """Return a list of datetime.date objects transformed from the data_input list which contains
    dates in the type string
    """

    # Set initial lists

    collect_all_dates = []
    return_lst = []

    # Append the dates only from the input data list

    for date in data_input[0]:
        collect_all_dates.append(date)

    # Split the month, day and year into strings

    collect_all_dates = [str.split(date) for date in collect_all_dates]

    # Remove the punctuation in the date strings

    for date in collect_all_dates:
        for word in date:
            for char in word:
                if all([char in PUNCTUATION]):
                    date.remove(word)
                    blank = str.replace(word, char, '')
                    date.insert(1, blank)

    # Transform the string dates into datetime.date objects

    for date in collect_all_dates:
        transform = datetime.date(year=int(date[2]),
                                  month=MONTH_TO_INT[date[0]],
                                  day=int(date[1]))

        # Append to the return list
        return_lst.append(transform)

    return return_lst  # Return the list with the dates transformed into datetime.date objects


def remove_extra_dates(dt_lst: list) -> list:
    """Return the list containing the dates from the correct timeframe by removing
    the extra dates from the date range in dt_lst.
    """

    # Initialize the list variables

    collect_2020_lst = []
    collect_2021_lst = []

    # Collect 2020 dates within the timeframe

    for date in dt_lst:
        if date.year >= 2020:
            collect_2020_lst.append(date)

    # Collect 2021 dates within the timeframe

    for date in collect_2020_lst:
        if date.year < 2021 and date.month >= 4:
            collect_2021_lst.append(date)
        elif date.year == 2021:
            collect_2021_lst.append(date)

    # Remove the first date to match the timeframe of dataset 1 and 2
    collect_2021_lst.remove(datetime.date(2020, 4, 4))

    return collect_2021_lst  # Return the list with the correct timeframe


def remove_extra_nums(nums: list, remove_num_lst: list) -> list:
    """Return the list containing the correct number of death counts, which are the values from
    remove_num_lst removed from nums.
    """

    # Remove the amount of nums given to the function

    for num in remove_num_lst:
        nums.remove(num)

    return nums


def get_dataset_values() -> tuple:
    """Return a tuple of five lists, each containing the data from each of the processed datasets.
    """

    # Dataset 1 values
    counts_list = read_death_count_data()
    # Get death counts data as lists (per columns in the dataset)
    sum_list, mn_list, doh_list, cr_list = \
        counts_list[1], counts_list[2], counts_list[3], counts_list[4]

    # Dataset 2 values
    hosp_data = read_hospitalization_data()
    # Get hospitalization data as list
    hospitalizations_per_week = hosp_data['Hospitalizations'].values
    hosp_list = list(hospitalizations_per_week)

    # Values of the datasets combined
    return sum_list, mn_list, doh_list, cr_list, hosp_list


# if __name__ == '__main__':
#     import python_ta
#     import python_ta.contracts
#
#     python_ta.contracts.DEBUG_CONTRACTS = False
#     python_ta.contracts.check_all_contracts()
#
#     python_ta.check_all(config={
#         'allowed-io': ['read_death_count_data'],
#         'extra-imports': ['python_ta.contracts', 'csv', 'datetime', 'pandas'],
#         'max-line-length': 100,
#         'disable': ['R1705', 'C0200'],
#     }, output='pyta_report.html')
