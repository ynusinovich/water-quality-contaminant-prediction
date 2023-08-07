from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class ModelTrainer():
    """"""

    def __init__(self, TRACKING_SERVER_HOST):
        """"""
        self.TRACKING_SERVER_HOST = TRACKING_SERVER_HOST

    def load_data(self):
        """Load the data and create X and y"""
        df = pd.read_parquet("../data/df.parquet")
        self.train_df = pd.read_parquet("../data/train_df.parquet")
        self.val_df = pd.read_parquet("../data/val_df.parquet")
        self.test_df = pd.read_parquet("../data/test_df.parquet")
        self.y = "Methyl tert-butyl ether (MTBE)"
        self.X_col = [c for c in df.columns if c not in [y, "sample_date", "station_id"]]

    def objective(self, params):
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
            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")     
        return {'loss': rmse, 'status': STATUS_OK}

    def run_training(self):
        """"""
        dv = DictVectorizer()
        train_dicts = self.train_df[self.X_col].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        val_dicts = self.val_df[self.X_col].to_dict(orient='records')
        X_val = dv.transform(val_dicts)
        y_train = self.train_df[self.y].values
        y_val = self.val_df[self.y].values
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        mlflow.set_tracking_uri(f"http://{self.TRACKING_SERVER_HOST}:5000")
        mlflow.set_experiment("water-quality-prediction")
        mlflow.xgboost.autolog()
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
            fn=self.objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials()
            )


def train_model(TRACKING_SERVER_HOST):
    """Main function for model training."""
    model_trainer = ModelTrainer(TRACKING_SERVER_HOST)
    model_trainer.load_data()
    model_trainer.run_training()


if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "default"
    TRACKING_SERVER_HOST= "ec2-54-147-5-224.compute-1.amazonaws.com"

    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    train_model(TRACKING_SERVER_HOST)