from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import joblib
import io

app = FastAPI()
model = None

# Load model from S3 once at startup
@app.on_event("startup")
def load_model():
    global model
    s3 = boto3.client("s3")
    bucket = "lambda-code-bucket-ddd"
    key = "model/model.pkl"

    obj = s3.get_object(Bucket=bucket, Key=key)
    bytestream = io.BytesIO(obj["Body"].read())
    model = joblib.load(bytestream)

class IrisInput(BaseModel):
    data: list  # ideally validate shape too

@app.post("/predict")
def predict(input: IrisInput):
    try:
        preds = model.predict([input.data])
        return {"prediction": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
