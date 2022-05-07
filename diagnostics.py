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
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# Function to get model predictions
def model_predictions():
    # read the deployed model and a test dataset, calculate predictions
    features_list = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']

    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    modelname = 'trainedmodel.pkl'
    model = pickle.load(open(os.path.join(os.getcwd(), prod_deployment_path, modelname), 'rb'))

    predictions = list(model.predict(df_test[features_list]))
    return predictions  # return value should be a list containing all predictions


# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    means_list = list(df_test.mean())

    return means_list  # return value should be a list containing all summary statistics


def dataframe_nas():
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
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
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    print(outdated)
    with open(os.path.join(os.getcwd(), 'outdated.txt'), 'wb') as outfile:
        outfile.write(outdated)


if __name__ == '__main__':
    predict = model_predictions()
    print(predict)
    summary = dataframe_summary()
    print(summary)
    dataframe_nas = dataframe_nas()
    print(dataframe_nas)
    timing_process = execution_time()
    print(timing_process)
    outdated_packages_list()

