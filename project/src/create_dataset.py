import json
import os
import logging
import requests
import datetime
import pandas as pd
from prefect import flow, task, get_run_logger
from prefect_aws import S3Bucket
from common_functions import download_data, clean_data

logging.basicConfig(level=logging.INFO)


@task
def train_val_test_split():
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

    logger = get_run_logger()
    logger.info("Data splitting complete")

    return train_df, val_df, test_df


@flow
def create_dataset(y="Methyl tert-butyl ether (MTBE)"):
    """Main function for dataset creation."""
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    download_data()
    clean_data(y, "train")
    train_val_test_split()


if __name__ == "__main__":
    create_dataset()
