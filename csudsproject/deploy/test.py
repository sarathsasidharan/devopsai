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

spn = ServicePrincipalAuthentication("72f988bf-86f1-41af-91ab-2d7cd011db47", "2d78ca87-de7e-437c-bfaf-e75d38a81398", "UlsxrsSkkTl2JO1t4NKGcgdN3AEXxm2fgYkqITi7vfQ=")
ws = Workspace(auth = spn,
            workspace_name = "amlservices",
            subscription_id = "5c667bbb-a09e-4d96-bfe6-6659ade1e2cc",
            resource_group = "amlservices")
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