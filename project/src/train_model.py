import logging
import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import pandas as pd
from prefect import flow
from prefect_aws import S3Bucket

logging.basicConfig(level=logging.INFO)


class ModelTrainer():
    """Class for loading the clean dataset and running training."""

    def __init__(self, tracking_server_host, y):
        """Initialize the model trainer object."""
        self.tracking_server_host = tracking_server_host
        self.y = y

    def load_data(self):
        """Load the data and create X and y"""
        s3_bucket_block = S3Bucket.load("s3-bucket-example")
        s3_bucket_block.download_folder_to_path(from_folder="project/data", to_folder="../data")
        df = pd.read_parquet("../data/df.parquet")
        train_df = pd.read_parquet("../data/train_df.parquet")
        val_df = pd.read_parquet("../data/val_df.parquet")
        X_col = [c for c in df.columns if c not in [self.y, "sample_date", "station_id"]]
        return train_df, val_df, X_col

    def run_training(self):
        """Run the model training with an XGBoost model and a range of parameters."""
        train_df, val_df, X_col = self.load_data()
        dv = DictVectorizer()
        train_dicts = train_df[X_col].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        val_dicts = val_df[X_col].to_dict(orient='records')
        X_val = dv.transform(val_dicts)
        y_train = train_df[self.y].values
        y_val = val_df[self.y].values
        if not os.path.exists("../preprocessor/"):
            os.makedirs("../preprocessor/")
        with open("../preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        mlflow.set_tracking_uri(f"http://{self.tracking_server_host}:5000")
        mlflow.set_experiment("water-quality-prediction-2")
        mlflow.xgboost.autolog()
        def objective(params):
            with mlflow.start_run():
                mlflow.set_tag("model", "xgboost")
                mlflow.log_params(params)
                booster = xgb.train(
                    params=params,
                    dtrain=train,
                    num_boost_round=1000,
                    evals=[(valid, 'validation')],
                    early_stopping_rounds=50
                    )
                y_pred = booster.predict(valid)
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")
            return {'loss': rmse, 'status': STATUS_OK}
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:linear',
            'seed': 42
            }
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials()
            )
        return best_result

@flow
def train_model(tracking_server_host="ec2-3-90-105-109.compute-1.amazonaws.com",
                y="Methyl tert-butyl ether (MTBE)"):
    """Main function for model training."""
    os.environ["AWS_PROFILE"] = "default"
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    model_trainer = ModelTrainer(tracking_server_host, y)
    model_trainer.load_data()
    model_trainer.run_training()


if __name__ == "__main__":
    TRACKING_SERVER_HOST = "ec2-3-90-105-109.compute-1.amazonaws.com"
    y = "Methyl tert-butyl ether (MTBE)"
    train_model(TRACKING_SERVER_HOST, y)
