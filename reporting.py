import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    features_list = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target = 'exited'

    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    modelname = 'trainedmodel.pkl'
    model = pickle.load(open(os.path.join(os.getcwd(), prod_deployment_path, modelname), 'rb'))
    predictions = model.predict(df_test[features_list])
    target = df_test.pop(target)
    cf_matrix = confusion_matrix(target, predictions)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig(os.path.join(os.getcwd(), prod_deployment_path, 'confusionmatrix.png'))


if __name__ == '__main__':
    score_model()
