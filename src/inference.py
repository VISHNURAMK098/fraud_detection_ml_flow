# FastAPI Serving
from fastapi import FastAPI
from pydantic import BaseModel
import boto3

app = FastAPI()

class IrisInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input: IrisInput):
    try:
        s3 = boto3.client("s3")
        model_obj = s3.get_object(Bucket="lambda-code-bucket-ddd", Key="model/model.pkl")
        preds = model_obj.predict([input.data])
        return {"prediction": preds.tolist()}
    except Exception as e:
        return {"prediction": e}
