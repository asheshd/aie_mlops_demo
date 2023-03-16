import os
import pickle
import numpy as np
import pandas as pd

from modelstore import ModelStore

from sklearn.metrics import recall_score, precision_score
import json

filename = "aie_model.sav"
X_test = np.genfromtxt("data/processed/test_features.csv")
y_test = np.genfromtxt("data/processed/test_labels.csv")


model_store = ModelStore.from_aws_s3("iiscdvc")
domain_name = "aie_domain"

model_path = model_store.download(
   local_path=".",
   domain=domain_name
)


#  Model loading for prediction
model = pickle.load(open(os.path.join("models", filename), 'rb'))


y_pred = model.predict(X_test)
acc = model.score (X_test, y_test)


# Actual value (y_test) vs Predicted Value (y_score)

prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# Get the loss
loss = model.loss_curve_
pd.DataFrame(loss, columns=["loss"]).to_csv("reports/loss.csv", index=False)

with open("reports/metrics.json", 'w') as outfile:
    json.dump({"accuracy": acc, "precision": prec, "recall": rec}, outfile)



