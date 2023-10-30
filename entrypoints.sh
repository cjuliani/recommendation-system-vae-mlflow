#!/bin/bash

# Check if the SQLite database file exists, and if not, create it
python mlflow_db.py
wait

# Start the MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///sqlite/mlflow.db --default-artifact-root /mlruns
