from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnostics
from diagnostics import model_predictions, dataframe_summary, dataframe_nas, execution_time, outdated_packages_list
from scoring import score_model
# import predict_exited_from_saved_model
import json
import os

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

modelname = 'trainedmodel.pkl'
prediction_model = pickle.load(open(os.path.join(os.getcwd(), prod_deployment_path, modelname), 'rb'))


def read_pandas(filename):
    df = pd.read_csv(filename)
    return df


@app.route("/", methods=['GET'])
def health():
    if request.method == "GET":
        return jsonify({"status": "ok"})


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    if request.method == "POST":
        filename = request.json['filepath']
        data_pd = read_pandas(filename)
        predictions = model_predictions(data_pd)
        return {"Prediction": str(predictions)}  # add return value for prediction outputs


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    if request.method == "GET":
        scoring = score_model()
        return {'F1 Score': scoring}  # add return value (a single F1 score number)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary():
    # check means, medians, and modes for each column
    if request.method == "GET":
        return {'Summary Stats': dataframe_summary()}  # return a list of all calculated summary statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    if request.method == "GET":
        time = execution_time()
        na_stats = dataframe_nas()
        out_list = outdated_packages_list()
        return {"Timing": time, "NA_stats": na_stats, "Outdated": out_list}     # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
