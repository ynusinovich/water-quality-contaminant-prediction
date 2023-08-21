import datetime
from dateutil.relativedelta import relativedelta
import time
import logging
import os
import pandas as pd
from psycopg2 import connect
from prefect import task, flow, get_run_logger

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, \
                              DatasetCorrelationsMetric

from common_functions import download_data, clean_data
from inference import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 1

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    target_prediction_correlation float
    )
"""


@task(retries=2, retry_delay_seconds=2, name="prep db")
def prep_db():
    """Creates database for storing metrics."""
    with connect("host=localhost port=5432 user=postgres password=example") as conn:
        cursor = conn.cursor()
        db_name = 'test'
        cursor.execute(
                        f"SELECT datname FROM pg_database WHERE datname = '{db_name}'"
                        )
        db_exists = cursor.fetchone() is not None
    if not db_exists:
        conn = connect(
                       user='postgres',
                       password='example',
                       host='localhost',
                       port='5432'
                      )
        conn.autocommit = True
        cursor = conn.cursor()
        sql = f"CREATE DATABASE {db_name};";
        cursor.execute(sql)
        conn.close()
    with connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_statement)


@task(retries=2, retry_delay_seconds=2, name="calculate metrics")
def calculate_metrics(curr, i,
                      tracking_server_host, stage, model_name,
                      y, begin, column_mapping, report,
                      df, model, dv):
    """Calculates the metrics and loads them into Postgres."""
    reference_data = df[df["sample_date"] < begin]
    start_date = begin + relativedelta(months=i)
    end_date = begin + relativedelta(months=i + 1)
    current_data = df[(df["sample_date"] >= start_date) & (df["sample_date"] < end_date)]

    inference_pipeline_ref = InferencePipeline(tracking_server_host, stage,
                                               model_name, y, datetime.date(1900, 1, 1), begin)
    reference_data["pred"] = inference_pipeline_ref.run_pred(reference_data, model, dv)[0]
    inference_pipeline_cur = InferencePipeline(tracking_server_host, stage,
                                               model_name, y, start_date, end_date)
    current_data["pred"] = inference_pipeline_cur.run_pred(current_data, model, dv)[0]

    report.run(reference_data=reference_data, current_data=current_data,
               column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    if not pd.isnull(result['metrics'][3]["result"]["current"]["stats"]["pearson"]["target_prediction_correlation"]):
        target_prediction_correlation = result['metrics'][3]["result"]["current"]["stats"]["pearson"]["target_prediction_correlation"]
    else:
        target_prediction_correlation = 0

    curr.execute(
                 "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, target_prediction_correlation) values (%s, %s, %s, %s, %s)",
                 (start_date, prediction_drift, num_drifted_columns, share_missing_values, target_prediction_correlation)
                )

@flow
def batch_monitoring_backfill(tracking_server_host = "ec2-3-90-105-109.compute-1.amazonaws.com",
                              stage = "Production",
                              model_name = "water-quality-predictor-3",
                              y = "Methyl tert-butyl ether (MTBE)"):
    """Generates drift, correlations, and missing values for the months of 2019 compared to past months."""
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    download_data()
    numerical_list = clean_data(y)
    begin = datetime.date(2019, 1, 1)
    month_iterations = 12
    column_mapping = ColumnMapping(
        prediction='pred',
        numerical_features=numerical_list,
        target=None
    )
    report = Report(metrics=[
                             ColumnDriftMetric(column_name='pred'),
                             DatasetDriftMetric(),
                             DatasetMissingValuesMetric(),
                             DatasetCorrelationsMetric()
                            ]
                   )
    prep_db()
    inference_pipeline = InferencePipeline(tracking_server_host, stage,
                                           model_name, y, None, None)
    df = inference_pipeline.load_data()
    model, dv = inference_pipeline.load_model_and_dv()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=2)
    with connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        conn.autocommit = True
        for i in range(0, month_iterations):
            with conn.cursor() as curr:
                calculate_metrics(curr, i,
                                  tracking_server_host, stage, model_name,
                                  y, begin, column_mapping, report,
                                  df, model, dv)
            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=1)
            # logger = get_run_logger()
            # logger.info("Data sent")

if __name__ == '__main__':
    batch_monitoring_backfill()
