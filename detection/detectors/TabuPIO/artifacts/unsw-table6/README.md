# Trained UNSW classifier

`model-0.joblib` contains the RBF OC-SVM, scaler and the paper’s 15-feature
subset (Table 6). Trained with scikit-learn 1.8.0; nu=0.001.

```python
import joblib

artifact = joblib.load('detection/detectors/TabuPIO/artifacts/unsw-table6/model-0.joblib')
model = artifact['model']
predictions = model.predict(records[model.feature_names].to_numpy(), variant='classifier_only')
```

Inputs are unscaled numeric UNSW flow features in `model.feature_names` order.
Encoding: proto tcp=0/udp=1; service dns=0; state CON=0/FIN=1/INT=2/REQ=3.
The embedded scaler applies normalization.

On 47,294 held-out records: TP=39,635, FP=3, TN=7,490, FN=166.
This reconstructs the classifier using the published subset. Optimizer
selection and the packet-length hybrid remain unverified; raw collector
packets do not supply these flow features.

Detailed training outputs stay under the ignored `results/` directory.
