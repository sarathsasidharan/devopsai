import argparse
import os,sys
import numpy as np
import gzip
import numpy as np
import struct

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from azureml.core import Run

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


# Args 
# 1) The location of the data files (from datastore)
# 2) Regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

data_folder = os.path.join(args.data_folder, 'mnist')
print('Data folder:', data_folder)

#Generate splits
print("Split up into train and test data")
x_train = load_data(data_folder+'/train-images.gz', False) / 255.0
y_train = load_data(data_folder+'/train-labels.gz', True).reshape(-1)
x_test = load_data(data_folder+'/test-images.gz', False) / 255.0
y_test = load_data(data_folder+'/test-labels.gz', True).reshape(-1)

# get hold of the current run
run = Run.get_context()

print('Train a logistic regression model with regularizaion rate of', args.reg)
clf = LogisticRegression(C=1.0/args.reg, random_state=42)
clf.fit(x_train, y_train)

print('Predict the test set')
y_hat = clf.predict(x_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

run.log('regularization rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
# File saved in the outputs folder is automatically uploaded into experiment record
print("File saved in the outputs folder is automatically uploaded into experiment record")
joblib.dump(value=clf, filename='outputs/csu_sklearn_mnist_model.pkl')

#Register the model in the Model Managment of azure ml service workspace
model = run.register_model(model_name='csu_sklearn_mnist', model_path='outputs/csu_sklearn_mnist_model.pkl')
print(model.name, model.id, model.version, sep = '\t')



