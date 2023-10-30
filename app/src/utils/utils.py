import os
import sqlite3
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt

from .metrics import NDCGMetric, RecallMetric
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


def create_or_set_path(path: str):
    """Creates directory and subdirectories and returns related
    path."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def late_logging(model, sampler, model_name, features, files_path, **kwargs):
    """Logs MLflow information while training."""
    # Define mlflow client
    client = MlflowClient()

    # Infer signature of the model
    signature = infer_signature(
        model_input=features[:1],
        model_output=model.predict(features[:1]))

    # Log the model variables
    mlflow.keras.log_model(
        model=model,
        registered_model_name=model_name,
        artifact_path='model',
        signature=signature,
        custom_objects={
            'RecallMetric': RecallMetric,
            'NDCGMetric': NDCGMetric
        }
    )

    mlflow.keras.log_model(
        model=sampler,
        registered_model_name=model_name+'_sampler',
        artifact_path='sampler'
    )

    # Log the inputs and summary which can be analyzed in MLflow UI
    mlflow.log_input(mlflow.data.from_numpy(features), context='features')

    with open(files_path + 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Log model-related files as artifacts
    mlflow.log_artifact(
        local_path=files_path,
        artifact_path="model",
    )

    # Set model tags related to runs and experiments via MLflow
    client.set_model_version_tag(
        name=model_name,
        stage='None',
        key='run_id',
        value=kwargs['run_id'])

    client.set_model_version_tag(
        name=model_name,
        stage='None',
        key='experiment_id',
        value=kwargs['experiment_id'])

    client.set_model_version_tag(
        name=model_name,
        stage='None',
        key='experiment_name',
        value=kwargs['experiment_name'])


def save_histogram(metrics_data: dict, save_path: str):
    """Saves histogram of metric values."""
    # Extract metric names and values from the dictionary
    metric_names = list(metrics_data.keys())
    metric_values = list(metrics_data.values())

    # Create a bar plot (histogram)
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    plt.bar(metric_names, metric_values, color='blue')

    # Customize plot labels and title
    plt.xlabel('Metric Names')
    plt.ylabel('Metric Values')
    plt.title('Histogram of Metrics')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Save the figure to the specified file path (without displaying it)
    plt.tight_layout()
    plt.savefig(save_path)

    # Close the plot
    plt.close()


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def collect_model_path(name: str, version: int) -> str:
    db_connection = sqlite3.connect('sqlite/mlflow.db')
    cursor = db_connection.cursor()

    # Execute SQL query to collect model source path
    query = f"SELECT source FROM model_versions WHERE name='{name}' AND version='{version}'"
    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    db_connection.close()

    return results[0][0]


def clean_up_backend_database():
    # Create a connection to the SQLite database
    db_connection = sqlite3.connect('sqlite/mlflow.db')

    # Create a cursor to execute SQL queries
    cursor = db_connection.cursor()

    # List of tables to delete data from
    tables_to_clear = [
        'runs', 'experiments', 'metrics', 'params',
        'tags', 'model_version_tags', 'registered_models',
        'model_versions', 'inputs', 'input_tags', 'datasets',
        'latest_metrics']

    # Iterate through the tables and delete all data
    for table in tables_to_clear:
        cursor.execute(f"DELETE FROM {table}")

    # Commit the changes and close the database connection
    db_connection.commit()
    db_connection.close()
