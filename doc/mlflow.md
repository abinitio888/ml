- Set up the mflow tracking_url
cd ml/

mlflow server --backend-store-uri sqlite:///mlflow_model.db --host 0.0.0.0
--default-artifact-root ./mlruns

- mlflow ui
cd ml/
mlflow ui --backend-store-uri sqlite:///mlflow_model.db
