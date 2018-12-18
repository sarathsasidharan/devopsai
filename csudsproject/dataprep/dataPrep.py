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

# Variables
tenant_id="<Enter Your Tenant Id>"
app_id="<Application Id of the SPN you Create>"
app_key= "<Key for the SPN>"
workspace="<Name of your workspace>"
subscription_id="<Subscription id>"
resource_grp="<Name of your resource group where aml service is created>"
experiment_name = '<Name of your experiment>'

# Print AML Version
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Point file to conf directory containing details for the aml service
spn = ServicePrincipalAuthentication(tenant_id,app_id,app_key)
ws = Workspace(auth = spn,
            workspace_name = workspace,
            subscription_id = subscription_id,
            resource_group = resource_grp)
print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')

# Create a new experiment
print("Starting to create new experiment")

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










