import azureml
from azureml.core import Workspace, Run
from azureml.core import Experiment
import os
import urllib.request
import sys, os
sys.path.append('../')
from services.CreateAmlCompute import CreateAmlCompute
from util.LoadData import LoadData
from azureml.core.authentication import ServicePrincipalAuthentication

# Print AML Version
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Point file to conf directory containing details for the aml service
spn = ServicePrincipalAuthentication("72f988bf-86f1-41af-91ab-2d7cd011db47", "2d78ca87-de7e-437c-bfaf-e75d38a81398", "UlsxrsSkkTl2JO1t4NKGcgdN3AEXxm2fgYkqITi7vfQ=")
ws = Workspace(auth = spn,
            workspace_name = "amlservices",
            subscription_id = "5c667bbb-a09e-4d96-bfe6-6659ade1e2cc",
            resource_group = "amlservices")
print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')

# Create a new experiment
print("Starting to create new experiment")
experiment_name = 'csu-image-recognition'
exp = Experiment(workspace=ws, name=experiment_name)

# Create / Reuse AML Compute
print("Create a new AML Compute , if exists will reuse")
aml_compute = CreateAmlCompute(ws)
aml_compute.create_aml_compute(ws)

#Load Data and generate splits
load_data = LoadData()
print("Started to download data from source")
load_data.download_data()
print("Split up into train and test data")
x_train,y_train,x_test,y_test= load_data.train_test_split()

#Upload Data to cloud / onto the ws default store , you could use your own ADLS if required
print("Started uploading data to blob storage")
load_data.load_data_to_blob(ws)










