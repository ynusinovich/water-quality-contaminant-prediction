# MLOps Zoomcamp from DataTalks.club: Final Project
## Predicting Concentrations of Expensive-to-Measure Water Quality Contaminants
### MLFlow server connection
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://mlflow:POSTGRES_PASSWORD@mlflow-database.cshyhxremkby.us-east-1.rds.amazonaws.com:5432/mlflow_db --default-artifact-root s3://mlops-zoomcamp-2023/project
### Prefect setup
prefect cloud login
prefect project init
prefect worker start -p project-pool -t process
prefect deploy project/src/create_dataset.py:create_dataset -n 'project-deployment-data' -p project-pool
prefect deploy project/src/train_model.py:train_model -n 'project-deployment-train' -p project-pool
prefect deploy project/src/register.py:register -n 'project-deployment-register' -p project-pool
prefect deploy project/src/inference.py:inference -n 'project-deployment-inference' -p project-pool
prefect deploy project/src/metrics_calculation.py:batch_monitoring_backfill -n 'project-deployment-metrics' -p project-pool
### Building Services for Monitoring
docker-compose up --build
### Grafana default username and password:
admin/admin
### Monitoring Postgres default server, username, password, and database:
db/postgres/example/test
