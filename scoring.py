from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model
    # relative to the test data
    # it should write the result to the latestscore.txt file
    features_list = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target = 'exited'

    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    modelname = 'trainedmodel.pkl'
    model = pickle.load(open(os.path.join(os.getcwd(), model_path, modelname), 'rb'))

    predictions = model.predict(df_test[features_list])

    f1_score = metrics.f1_score(df_test[target], predictions, average=None)

    latestscore = open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'w')
    latestscore.write(str(f1_score[1]) + "\n")
    latestscore.close()


if __name__ == '__main__':
    score_model()
