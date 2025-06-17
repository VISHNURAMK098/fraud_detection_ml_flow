import boto3
import pandas as pd
from io import StringIO

def load_csv_from_s3(bucket_name: str, object_key: str) -> pd.DataFrame:
    """Load a CSV file from an S3 bucket into a Pandas DataFrame."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    content = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(content))
