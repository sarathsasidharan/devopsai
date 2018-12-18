from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

class CreateAmlCompute:

    def __init__(self, ws):
        self.ws = ws
   
    def create_aml_compute(self,ws):
        # choose a name for your cluster
        print("Creating new AML Compute")
        compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
        compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
        compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)
        print(" AML Compute "+compute_name+" min nodes "+str(compute_min_nodes)+" compute max nodes "+str(compute_max_nodes))
        # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
        vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

        if compute_name in ws.compute_targets:
            compute_target = ws.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print('found compute target. just use it. ' + compute_name)
        else:
            print('creating a new compute target...')
            provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = compute_min_nodes, 
                                                                max_nodes = compute_max_nodes)
            # create the cluster
            print("Starting to create ACI Compute cluster")
            compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
            # can poll for a minimum number of nodes and for a specific timeout. 
            # if no min node count is provided it will use the scale settings for the cluster
            compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        return compute_target