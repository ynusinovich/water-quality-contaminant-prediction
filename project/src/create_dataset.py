import json
import os
import logging
import requests
import datetime
import pandas as pd
from prefect import flow
from prefect_aws import S3Bucket
from common_functions import clean_data

logging.basicConfig(level=logging.INFO)


class DatasetCreator():
    """Class to download and clean dataset."""

    def __init__(self, y):
        """Initialize the dataset creator."""
        self.y = y

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

    def train_val_test_split(self):
        """Split data along time axis into training, validation, and test."""
        df = pd.read_parquet("../data/df.parquet")
        
        train_max = datetime.date(2013,1,1)
        val_max = datetime.date(2016,1,1)
        test_max = datetime.date(2019,1,1)
        df_model = df[df["sample_date"] < test_max]
        train_df = df_model[df_model["sample_date"] < train_max]
        val_df = df_model[(df_model["sample_date"] < val_max) & (df_model["sample_date"] > train_max)]
        test_df = df_model[df_model["sample_date"]>val_max]

        train_df.to_parquet("../data/train_df.parquet")
        val_df.to_parquet("../data/val_df.parquet")
        test_df.to_parquet("../data/test_df.parquet")

        s3_bucket_block = S3Bucket.load("s3-bucket-example")
        s3_bucket_block.put_directory(local_path="../data", to_path="project/data")


@flow
def create_dataset(download=True, clean=True, split=True, y="Methyl tert-butyl ether (MTBE)"):
    """Main function for dataset creation."""
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    dataset_creator = DatasetCreator(y)
    if download:
        dataset_creator.download_data()
    if clean:
        clean_data(y)
    if split:
        dataset_creator.train_val_test_split()


if __name__ == "__main__":
    download = True
    clean = True
    split = True
    y = "Methyl tert-butyl ether (MTBE)"
    create_dataset(download, clean, split, y)
