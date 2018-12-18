from azureml.train.estimator import Estimator
from azureml.core import Workspace, Run
from azureml.core import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication

def trigger_training_job(experiment_name,compute_name,script_folder):
    print("Starting trigger engine")
    # Start creating 
    spn = ServicePrincipalAuthentication("72f988bf-86f1-41af-91ab-2d7cd011db47", "2d78ca87-de7e-437c-bfaf-e75d38a81398", "UlsxrsSkkTl2JO1t4NKGcgdN3AEXxm2fgYkqITi7vfQ=")
    ws = Workspace(auth = spn,
               workspace_name = "amlservices",
               subscription_id = "5c667bbb-a09e-4d96-bfe6-6659ade1e2cc",
               resource_group = "amlservices")

    #ws = Workspace.from_config(path="../conf/config.json")
    ds = ws.get_default_datastore()
    print(ds.datastore_type, ds.account_name, ds.container_name)
    exp = Experiment(workspace=ws, name=experiment_name)
    compute_target = ws.compute_targets[compute_name]
    script_params = {
        '--data-folder': ds.as_mount(),
        '--regularization': 0.8
    }
    est = Estimator(source_directory=script_folder,
            script_params=script_params,
            compute_target=compute_target,
            entry_script='train.py',
            conda_packages=['scikit-learn'])
    print("Submitting Runs to AML compute "+compute_name)
    run = exp.submit(config=est)
    run.wait_for_completion(show_output=True) # specify True for a verbose log


def main():
    script_folder="../modelling"
    compute_name="cpucluster"
    experiment_name="csu-image-recognition"
    trigger_training_job(experiment_name,compute_name,script_folder)

if __name__ == "__main__":
    main()

