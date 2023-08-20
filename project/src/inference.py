# pylint: disable=missing-module-docstring
import logging
import pickle
import os
import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
import mlflow
# pylint: disable=import-error
from prefect import flow, get_run_logger
from common_functions import download_data, clean_data

logging.basicConfig(level=logging.INFO)


class InferencePipeline():
    """Class to perform inference."""

    def __init__(self, tracking_server_host, stage, model_name, y):
        """Initialize the inference pipeline"""
        self.tracking_server_host = tracking_server_host
        self.stage = stage
        self.model_name = model_name
        self.y = y

    def create_inf_df(self):
        """Create inference dataframe from latest data."""
        df = pd.read_parquet("../data/df.parquet")
        X_values_to_keep = [X for X in df.columns if X != self.y]
        df[X_values_to_keep].dropna(inplace=True)
        test_max = datetime.date(2019,1,1)
        inf_df = df[df["sample_date"] > test_max]
        return inf_df

    def run_pred(self, inf_df):
        """Load mlflow model and artifacts, process inference data, and run prediction."""
        mlflow.set_tracking_uri(f"http://{self.tracking_server_host}:5000")
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{self.stage}")
        run_id = model.metadata.run_id
        mlflow.artifacts.download_artifacts(run_id=run_id,
                                            artifact_path="preprocessor",
                                            dst_path="../")
        if not os.path.exists("../preprocessor/"):
            os.makedirs("../preprocessor/")
        with open("../preprocessor/preprocessor.b", "rb") as f_in:
            dv = pickle.load(f_in)

        X_col = [c for c in inf_df.columns if c not in [self.y, "sample_date", "station_id"]]
        X_dicts = inf_df[X_col].to_dict(orient='records')
        X_inf = dv.transform(X_dicts)
        y_inf = inf_df[self.y]

        pred = model.predict(X_inf)
        return pred, y_inf

@flow
def inference(tracking_server_host="ec2-3-90-105-109.compute-1.amazonaws.com",
                       stage="Production",
                       model_name = "water-quality-predictor-3",
                       y="Methyl tert-butyl ether (MTBE)"):
    """Main function for inference pipeline with new water quality data."""
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    download_data()
    clean_data(y)
    inference_pipeline = InferencePipeline(tracking_server_host, stage,
                                           model_name, y)
    inf_df = inference_pipeline.create_inf_df()
    pred, y_inf = inference_pipeline.run_pred(inf_df)
    notnull_y_ind = [index for index, value in enumerate(y_inf) if pd.notnull(value)]
    notnull_y = y_inf[y_inf.notnull()]
    notnull_pred = pred[notnull_y_ind]
    rmse = mean_squared_error(notnull_y, notnull_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"rmse = {rmse}")
    inf_df["pred"] = pred
    inf_df.to_csv("../data/results.csv")
    return rmse, pred


if __name__ == "__main__":
    inference()