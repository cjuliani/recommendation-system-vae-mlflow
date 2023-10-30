import mlflow
from mlflow.store.db import utils


if __name__ == '__main__':
    # Update database to MLflow format
    engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(
        db_uri='sqlite:///sqlite/mlflow.db'
    )
    utils._initialize_tables(engine)
