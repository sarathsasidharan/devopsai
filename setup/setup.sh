#!/bin/bash
python --version
pip install azure-cli==2.0.46
pip install --upgrade azureml-sdk[notebooks,automl]
pip install scikit-learn==0.19.1