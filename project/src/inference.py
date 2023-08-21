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

    def __init__(self, tracking_server_host, stage,
                 model_name, y, inf_min, inf_max):
        """Initialize the inference pipeline"""
        self.tracking_server_host = tracking_server_host
        self.stage = stage
        self.model_name = model_name
        self.y = y
        self.inf_min = inf_min
        self.inf_max = inf_max

    def load_data(self):
        """Read data from local."""
        df = pd.read_parquet("../data/df.parquet")
        return df
    
    def load_model_and_dv(self):
        """Load model and dictionary vectorizer from MLFlow server."""
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
        return model, dv

    def run_pred(self, inf_df, model, dv):
        """Load mlflow model and artifacts, process inference data, and run prediction."""
        inf_df = inf_df[(inf_df["sample_date"] >= self.inf_min) & (inf_df["sample_date"] < self.inf_max)]
        X_col = [c for c in inf_df.columns if c not in [self.y, "sample_date", "station_id"]]
        X_dicts = inf_df[X_col].to_dict(orient='records')
        X_inf = dv.transform(X_dicts)
        y_inf = inf_df[self.y]

        pred = model.predict(X_inf)
        return pred, y_inf, inf_df

@flow
def inference(tracking_server_host="ec2-3-90-105-109.compute-1.amazonaws.com",
                       stage="Production",
                       model_name = "water-quality-predictor-3",
                       y="Methyl tert-butyl ether (MTBE)",
                       download_and_clean=True,
                       inf_min=datetime.date(2019,1,1),
                       inf_max=datetime.date.today()):
    """Main function for inference pipeline with new water quality data."""
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    if download_and_clean:
        download_data()
        clean_data(y)
    inference_pipeline = InferencePipeline(tracking_server_host, stage,
                                           model_name, y, inf_min, inf_max)
    inf_df, model, dv = inference_pipeline.load_data_and_model()
    pred, y_inf, inf_df = inference_pipeline.run_pred(inf_df, model, dv)
    notnull_y_indexes = [index for index, value in enumerate(y_inf) if pd.notnull(value)]
    notnull_y_inf = y_inf[y_inf.notnull()]
    notnull_pred = pred[notnull_y_indexes]
    rmse = mean_squared_error(notnull_y_inf, notnull_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"rmse = {rmse}")
    inf_df["pred"] = pred
    return rmse, pred


if __name__ == "__main__":
    inference()
