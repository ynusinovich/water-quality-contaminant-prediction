import logging
import os
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow
from prefect import flow, get_run_logger

logging.basicConfig(level=logging.INFO)


@flow
def register(tracking_server_host="ec2-3-90-105-109.compute-1.amazonaws.com",
                   stage="Production",
                   experiment_name="water-quality-prediction-2",
                   experiment_ids='3',
                   model_name = "water-quality-predictor-4"):
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    mlflow.set_tracking_uri(f"http://{tracking_server_host}:5000")
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=f"http://{tracking_server_host}:5000")
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.val_rmse ASC"]
    )
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = model_name
    mlflow.register_model(model_uri=model_uri, name=model_name)
    model_version = 1
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
        archive_existing_versions=True
    )
    logger = get_run_logger()
    logger.info("Data cleaning complete")
    return None


if __name__ == "__main__":
    register()
