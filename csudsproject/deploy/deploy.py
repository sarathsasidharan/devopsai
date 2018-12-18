import azureml
from azureml.core import Workspace, Run
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage


# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

from azureml.core import Workspace
from azureml.core.model import Model

print("Starting to download the model file")
spn = ServicePrincipalAuthentication("72f988bf-86f1-41af-91ab-2d7cd011db47", "2d78ca87-de7e-437c-bfaf-e75d38a81398", "UlsxrsSkkTl2JO1t4NKGcgdN3AEXxm2fgYkqITi7vfQ=")
ws = Workspace(auth = spn,
            workspace_name = "amlservices",
            subscription_id = "5c667bbb-a09e-4d96-bfe6-6659ade1e2cc",
            resource_group = "amlservices")
print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')
model=Model(ws,'csu_sklearn_mnist')
model.download(target_dir = '.')
import os 
# verify the downloaded model file
os.stat('./csu_sklearn_mnist_model.pkl')
print("Downloaded Model File")

print("Writing Conda File")
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")


with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
print("Finished Writing Conda File")

print("Defined deploy configuration for ACI")
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "MNIST",  "method" : "sklearn"}, 
                                               description='Predict MNIST with sklearn')    


print("Configuring Image")
# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")

service = Webservice.deploy_from_model(workspace=ws,
                                       name='sklearn-mnist-svc',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)