import asyncio
import bentoml
import pandas as pd
import numpy as np

from bentoml.io import PandasDataFrame, JSON
from sample import sample_input

fraud_model_tiny_runner = bentoml.xgboost.get("ieee-fraud-detection-tiny:latest").to_runner()
fraud_model_small_runner = bentoml.xgboost.get("ieee-fraud-detection-small:latest").to_runner()
fraud_model_large_runner = bentoml.xgboost.get("ieee-fraud-detection-large:latest").to_runner()


svc = bentoml.Service("fraud_detection_graph", runners=[fraud_model_tiny_runner, fraud_model_small_runner, fraud_model_large_runner]) 

input_spec = PandasDataFrame.from_sample(sample_input)

@svc.api(input=input_spec, output=JSON())
def is_fraud(input_df: pd.DataFrame):
    results = fraud_model_large_runner.predict_proba.run(input_df)
    prediction = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {
        "is_fraud": bool(prediction),
        "is_fraud_prob": results[:,1]
    }

async def _is_fraud_async(
    runner: bentoml.Runner,
    input_df: pd.DataFrame,
):
    results = await runner.predict_proba.async_run(input_df)
    prediction = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {
        "is_fraud": bool(prediction),
        "is_fraud_prob": results[:,1]
    }

@svc.api(input=input_spec, output=JSON())
async def is_fraud_graph(input_df: pd.DataFrame):
    return await asyncio.gather(
        _is_fraud_async(fraud_model_tiny_runner, input_df),
        _is_fraud_async(fraud_model_small_runner, input_df),
        _is_fraud_async(fraud_model_large_runner, input_df),
    )
