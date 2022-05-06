from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


# function for deployment
def store_model_into_pickle(model):
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    #modelname = 'trainedmodel.pkl'
    source = os.path.join(os.getcwd(), model_path, model)
    target = os.path.join(os.getcwd(), prod_deployment_path, model)
    shutil.copyfile(source, target)

    lastestdata = 'ingestedfiles.txt'
    source = os.path.join(os.getcwd(), dataset_csv_path, lastestdata)
    target = os.path.join(os.getcwd(), prod_deployment_path, lastestdata)
    shutil.copyfile(source, target)

    lastestscore = 'latestscore.txt'
    source = os.path.join(os.getcwd(), model_path, lastestscore)
    target = os.path.join(os.getcwd(), prod_deployment_path, lastestscore)
    shutil.copyfile(source, target)


if __name__ == '__main__':
    store_model_into_pickle('trainedmodel.pkl')
