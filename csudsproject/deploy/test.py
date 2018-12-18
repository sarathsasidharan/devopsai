import requests
import json
import numpy as np
import gzip
import struct
import os
import urllib.request
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace, Run
from azureml.core.webservice import Webservice

# Define Vars < Change the vars>
tenant_id="<Enter Your Tenant Id>"
app_id="<Application Id of the SPN you Create>"
app_key= "<Key for the SPN>"
workspace="<Name of your workspace>"
subscription_id="<Subscription id>"
resource_grp="<Name of your resource group where aml service is created>"
experiment_name='<Name of your experiment you defined in dataprep.py>'

print("Starting trigger engine")
# Start creating 
# Point file to conf directory containing details for the aml service
spn = ServicePrincipalAuthentication(tenant_id,app_id,app_key)
ws = Workspace(auth = spn,
            workspace_name = workspace,
            subscription_id = subscription_id,
            resource_group = resource_grp)
print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')

service= Webservice(ws,'sklearn-mnist-svc')



#Function to load and  parse images 
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

os.makedirs('./data', exist_ok = True)  
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz') 
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')


X_test = load_data('./data/test-images.gz', False) / 255.0
y_test = load_data('./data/test-labels.gz', True).reshape(-1)


# send a random row from the test set to score
random_index = np.random.randint(0, len(X_test)-1)
input_data = "{\"data\": [" + str(list(X_test[random_index])) + "]}"
headers = {'Content-Type':'application/json'}

resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
#print("input data:", input_data)
print("label:", y_test[random_index])
print("prediction:", resp.text)