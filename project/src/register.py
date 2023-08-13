import logging
import os
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow

logging.basicConfig(level=logging.INFO)


def stage_model(tracking_server_host, stage):
    mlflow.set_tracking_uri(f"http://{tracking_server_host}:5000")
    mlflow.set_experiment("water-quality-prediction")
    client = MlflowClient(tracking_uri=f"http://{tracking_server_host}:5000")
    runs = client.search_runs(
        experiment_ids='1',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = "water-quality-predictor"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    model_version = 1
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
        archive_existing_versions=True
    )


if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "default"
    TRACKING_SERVER_HOST = "ec2-54-147-5-224.compute-1.amazonaws.com"
    stage = "Staging"

    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    stage_model(TRACKING_SERVER_HOST, stage)
