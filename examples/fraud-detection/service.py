import bentoml
import pandas as pd
import numpy as np

from bentoml.io import PandasDataFrame, JSON
from sample import sample_input

fraud_model_runner = bentoml.xgboost.get("ieee-fraud-detection-lg:latest").to_runner()


svc = bentoml.Service("fraud_detection", runners=[fraud_model_runner]) 

input_spec = PandasDataFrame.from_sample(sample_input)

@svc.api(input=input_spec, output=JSON())
def is_fraud(input_df: pd.DataFrame):
    results = fraud_model_runner.predict_proba.run(input_df)
    prediction = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {
        "is_fraud": bool(prediction),
        "is_fraud_prob": results[:,1]
    }
