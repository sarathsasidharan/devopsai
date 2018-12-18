from azureml.train.estimator import Estimator
from azureml.core import Workspace, Run
from azureml.core import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication

def trigger_training_job(compute_name,script_folder):

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
    trigger_training_job(compute_name,script_folder)

if __name__ == "__main__":
    main()

