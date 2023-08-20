import json
import logging
import os
import requests
import pandas as pd
import datetime
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow
from prefect import flow

logging.basicConfig(level=logging.INFO)


class InferencePipeline():
    """Class to perform inference."""

    def __init__(self, tracking_server_host, stage, model_name):
        """Initialize the inference pipeline"""
        self.tracking_server_host = tracking_server_host
        self.stage = stage
        self.model_name = model_name

    def download_data(self):
        """Download data from the source."""
        if not os.path.exists("../data/"):
            os.makedirs("../data/")
        with open("data_sources.json", "r", encoding="utf-8") as file:
            urls = json.load(file)
        url_field_results = urls["url_field_results"]
        url_lab_results = urls["url_lab_results"]
        for item in [("field_results.csv", url_field_results),
                        ("lab_results.csv", url_lab_results)]:
            try:
                response = requests.get(item[1], allow_redirects=True, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                logging.error("The request timed out. Try increasing the timeout value.")
            except requests.exceptions.RequestException as error:
                logging.error("An error occurred: %s", error)
            open(f'../data/{item[0]}', 'wb').write(response.content)
        logging.info("Data download complete")

    def create_inf_df(self):
        """Create inference dataframe from latest data."""
        df = pd.read_parquet("../data/df.parquet")
        test_max = datetime.date(2019,1,1)
        inf_df = df[df["sample_date"] > test_max]
        if not os.path.exists("../data/"):
            os.makedirs("../data/")
        inf_df.to_parquet("../data/inf_df.parquet")

        # s3_bucket_block = S3Bucket.load("s3-bucket-example")
        # s3_bucket_block.put_directory(local_path="../data", to_path="project/data")

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

def run_inference(tracking_server_host="ec2-3-90-105-109.compute-1.amazonaws.com",
                       stage="Production",
                       model_name = "water-quality-predictor-3"):
    inference_pipeline = InferencePipeline(tracking_server_host, stage, model_name)
    



if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "default"
    TRACKING_SERVER_HOST = "ec2-3-90-105-109.compute-1.amazonaws.com"
    stage = "Production"
    # experiment_name="water-quality-prediction-2"
    # experiment_ids='3'
    model_name = "water-quality-predictor-3"

    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    run_inference(TRACKING_SERVER_HOST, stage, model_name)