import pickle
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# Function to get model predictions
def model_predictions(dataset):
    # read the deployed model and a test dataset, calculate predictions
    features_list = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']

    modelname = 'trainedmodel.pkl'
    model = pickle.load(open(os.path.join(os.getcwd(), prod_deployment_path, modelname), 'rb'))

    predictions = list(model.predict(dataset[features_list]))
    return predictions  # return value should be a list containing all predictions


# Function to get summary statistics
def dataframe_summary():
    file = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    df_test = pd.read_csv(file)
    means_list = list(df_test.mean())
    median_list = list(df_test.median())
    std_list = list(df_test.std())

    return [means_list, median_list, std_list]  # return value should be a list containing all summary statistics


def dataframe_nas():
    file = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    df_test = pd.read_csv(file)
    nas = list(df_test.isna().sum())
    return [nas[i]/len(df_test.index) for i in range(len(nas))]


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    # training Timing :
    start_time = timeit.default_timer()
    os.system('python3 training.py')
    timing = timeit.default_timer() - start_time
    training_timing = timing

    # Ingestion Timing :
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing = timeit.default_timer() - start_time
    ingestion_timing = timing

    return [training_timing, ingestion_timing]  # return a list of 2 timing values in seconds


# Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    outdated = subprocess.check_output(['pip', 'list', '--outdated', '--format', 'columns'])
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'outdated.txt'), 'wb') as outfile:
        outfile.write(outdated)
    return outdated.decode("utf-8")


if __name__ == '__main__':
    dataframe_summary()
    dataframe_nas()
    execution_time()
    outdated_packages_list()
