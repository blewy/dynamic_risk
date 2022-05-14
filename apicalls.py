import requests
import os
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"
URL = "http://0.0.0.0:8000/"

# Call each API endpoint and store the responses
response1 = requests.post(url=URL + 'prediction',
                          json={'filepath': './ingesteddata/finaldata.csv'}).content  # put an API call here
response2 = requests.get(URL + 'scoring').content
response3 = requests.get(URL + 'summarystats').content
response4 = requests.get(URL + 'diagnostics').content

# combine all API responses
responses = {'response1': response1,
            'response2': response2,
            'response3': response3,
            'response4': response4} #combine responses here

# write the responses to your workspace
with open(os.path.join(os.getcwd(), 'apireturns.txt'), 'w') as outfile:
    for response in responses:
        print(responses[response], file=outfile)
        # outfile.write((responses))
