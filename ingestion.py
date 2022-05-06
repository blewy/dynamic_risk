import pandas as pd
# import numpy as np
import os
import json
from datetime import datetime

# 1Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    df_list = pd.DataFrame(columns=['corporation', 'lastmonth_activity',
                                    'lastyear_activity', 'number_of_employees', 'exited'])

    filenames = os.listdir(os.path.join(os.getcwd(), input_folder_path))
    filenames = [filename for filename in filenames if '.csv' in filename]  # removing all nin csv files

    datetimeobj = datetime.now()
    thetimenow = str(datetimeobj.year) + '/'+str(datetimeobj.month) + '/'+str(datetimeobj.day)

    allrecords = []

    for each_filename in filenames:
        sourcelocation = os.path.join(os.getcwd(), input_folder_path)
        df1 = pd.read_csv(os.path.join(sourcelocation, each_filename))
        df_list = df_list.append(df1)
        allrecords.append([sourcelocation, each_filename, len(df1.index), thetimenow])

    result = df_list.drop_duplicates()
    result.to_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'), index=False)

    ingestionfile = open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w')
    for element in allrecords:
        ingestionfile.write(str(element)+ "\n")

    ingestionfile.close()


if __name__ == '__main__':
    merge_multiple_dataframe()
