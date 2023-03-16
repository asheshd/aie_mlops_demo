from sklearn.neural_network import MLPClassifier

import os
import numpy as np
import pickle

from modelstore import ModelStore
import json

X_train = np.genfromtxt("data/processed/train_features.csv")
y_train = np.genfromtxt("data/processed/train_labels.csv")

model = MLPClassifier(random_state=0, max_iter=1)
model.fit (X_train, y_train)

# Manual Process

file_name = "aie_model.sav"
pickle.dump(model, open(os.path.join("models", file_name), 'wb'))

# Remote method

model_store = ModelStore.from_aws_s3("iiscdvc")

domain = "aie_domain"
out = model_store.upload(domain, model= model)

print(json.dumps(out, indent=4))
print("Model saved sucessfully in S3")


