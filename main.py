from src.data_loader import load_csv_from_s3
from src.preprocess import preprocess_data
from src.train import train_model
from src.logger import get_logger
import pandas as pd
import boto3
import json
import os
import io

logger = get_logger("fraud_detection")

def run_pipeline():
    logger.info("Starting fraud detection pipeline...")

    # Load data (local or from S3)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket="##add your bucket name", Key="upload/transaction_samples.csv")
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info("Data loaded from S3.")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    logger.info("Data preprocessing complete.")

    # Train and log
    metrics = train_model(X_train, X_test, y_train, y_test, scaler, logger)

    # Save metrics to S3
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket="your-bucket-name",
        Key="metrics/metrics.json",
        Body=json.dumps(metrics),
        ContentType="application/json"
    )
    logger.info("Metrics uploaded to S3.")

    return metrics

def lambda_handler(event, context): # GET /?key=metrics/metrics.json
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket="your-bucket-name", Key="metrics/metrics.json")
        metrics = json.loads(obj["Body"].read().decode("utf-8"))
        return {
            "statusCode": 200,
            "body": json.dumps(metrics),
            "headers": {"Content-Type": "application/json"}
        }
    except Exception as e:
        logger.error("Error in lambda_handler", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }

if __name__ == "__main__":
    run_pipeline()
