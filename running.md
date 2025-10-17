## backend:
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload


## mlflow
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root file://$(pwd)/orchestrations/mlflow_artifacts \
--host 0.0.0.0 \
--port 5050