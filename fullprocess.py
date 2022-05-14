import training
import scoring
import deployment
import diagnostics
import reporting
import os
import sys
import json
import pickle
import pandas as pd
from sklearn import metrics

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

ingested_folder_path = config['output_folder_path']
input_folder_path = config['input_folder_path']
model_path = os.path.join(config['output_model_path'])


# Check and read new data
# first, read ingestedfiles.txt

# get filenames
filenames = os.listdir(os.path.join(os.getcwd(), input_folder_path))
filenames = [filename for filename in filenames if '.csv' in filename]  # removing all nin csv files

# get filenames ingested
with open(os.path.join(os.getcwd(), ingested_folder_path, 'ingestedfiles.txt'), 'r') as f:
    read_data = f.readlines()

ingested_files = []  # List of ingested files
for line in read_data:
    file = ((line.split(',')[1]).strip()[1:-1])
    ingested_files.append(file)

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
# are the files contained on the list of ingested files?
is_subset = set(filenames).issubset(set(ingested_files))

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
# if not run ingestion.py
if not is_subset:
    os.system('python3 ingestion.py')

else:
    sys.exit()  # no new data = Stops the process

# Checking for model drift
# check whether the score from the deployed model is different from the score
# from the model that uses the newest ingested data

# get the latest score
if os.path.exists(os.path.join(os.getcwd(), model_path, 'latestscore.txt')):
    with open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'r') as f:
        score = f.readline()

        # get the latest model
        modelname = 'trainedmodel.pkl'
        model = pickle.load(open(os.path.join(os.getcwd(), model_path, modelname), 'rb'))

        features_list = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
        target = 'exited'

        df_test = pd.read_csv(os.path.join(ingested_folder_path, 'finaldata.csv'))
        predictions = model.predict(df_test[features_list])
        new_f1_score = metrics.f1_score(df_test[target], predictions, average='macro')

else:
    print("No previous scored is saved: Retraining the model")
    score = 0
    new_f1_score = 1



# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if float(score) < float(new_f1_score):
    print("Retraining")
    os.system('python3 training.py')
    print("Scoring")
    os.system('python3 scoring.py')

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
if float(score) < float(new_f1_score):
    print("Re-Deploy")
    os.system('python3 deployment.py')

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
if float(score) < float(new_f1_score):
    print("Update Diagnostics & Reporting for new model")
    os.system('python3 diagnostics.py')
    os.system('python3 reporting.py')
