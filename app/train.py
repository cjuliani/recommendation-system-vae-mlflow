import mlflow.keras
import argparse
import signal
import sys
import pickle
import pandas as pd
import tensorflow as tf
import os

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import custom_object_scope

from src.utils.loss import VAELoss, custom_loss_factory
from src.utils.metrics import NDCGMetric, RecallMetric
from src.utils.errors import handle_bool_parsing
from src.utils.utils import create_or_set_path, late_logging, strtobool, collect_model_path
from src.utils.callbacks import MLflowLoggingCallback
from src.utils.processing import split_data, remap_data, create_interaction_matrix
from src.data import DataGenerator
from src.model import build_model


parser = argparse.ArgumentParser(description="Train a machine learning model")
parser.add_argument("--experiment_name", type=str, default='0', help="Name of train experiment.")
parser.add_argument("--model_name", type=str, default='Unknown', help="Name of existing model to load.")
parser.add_argument("--model_version", type=int, default=0, help="Version of model loaded.")
bool_checker = lambda x: bool(strtobool(handle_bool_parsing(x, '--fine_tune')))
parser.add_argument("--fine_tune", default=False, type=bool_checker, help="If True, fine tune loaded model.")
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
parser.add_argument("--intermediate_dim", type=int, default=256, help="Training batch size.")
parser.add_argument("--latent_dim", type=int, default=64, help="Training batch size.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
parser.add_argument("--train_datio", type=float, default=0.95, help="Ratio of total data considered from training.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--minimum_lr", type=float, default=1e-5, help="Minimum learning rate to reach.")
parser.add_argument("--lr_decay_ratio", type=float, default=0.98, help="Learning rate decay factor.")
parser.add_argument("--minimum_interactions_per_item", type=int, default=60,
                    help="Minimum number of user interactions per individual item.")
parser.add_argument("--minimum_interactions_per_user", type=int, default=6,
                    help="Minimum number of interactions with items per user.")
args = parser.parse_args()



def handle_interrupt(signum, frame):
    # This function will be called when the script is
    # interrupted (e.g., Ctrl+C).
    print("Script interrupted. Logging model, inputs, and summary...")
    late_logging(
        model=model,
        sampler=sampler,
        features=x_train,
        model_name=args.model_name,
        run_id=active_run.info.run_id,
        experiment_id=experiment_id,
        experiment_name=args.experiment_name,
        files_path=filesdir)
    print("Logging complete. Exiting...")
    sys.exit(1)

# Register the signal handler for Ctrl+C (interrupt signal)
signal.signal(signal.SIGINT, handle_interrupt)

# Set local path to store artifacts related to an mlflow run.
filesdir = create_or_set_path("files/")

# Read movies data
#path = r'data\movielens_train.dat'
path = os.path.join('data', 'movielens_train.dat')
data = pd.read_csv(path, sep="::", names=["user", "item", "label", "time"], engine='python')

# Re-initialize item and user IDs so that they start from 0
train_data = data.drop(columns=['time'])  # time not analyzed
train_data = remap_data(train_data)

# Build user-item interaction matrix
inter_matrix = create_interaction_matrix(
    train_data=train_data,
    min_user_inter=args.minimum_interactions_per_user,
    min_items_inter=args.minimum_interactions_per_item)

original_dim = inter_matrix.shape[1]  # number of unique items kept

# Set random seed for TensorFlow and Numpy
random_seed = 42
tf.random.set_seed(random_seed)

# Split matrix data into train and validation sets
x_train, x_val, x_eval, num_steps_per_epoch = split_data(inter_matrix, args.batch_size, float(args.train_datio))

with open(filesdir+'x_eval.pkl', 'wb') as file:
    pickle.dump(x_eval, file)

with open(filesdir+'x_train.pkl', 'wb') as file:
    pickle.dump(x_train, file)

# Build model and batch generator for training
data_gen = DataGenerator(x_train, args.batch_size)
model, sampler = build_model(original_dim, args.intermediate_dim, args.latent_dim)

# Define callback functions
early_stop = EarlyStopping(
    monitor='loss',
    mode='min',
    verbose=1,
    patience=15)

lr_decay = ReduceLROnPlateau(  # adjust the learning rate during training
    monitor='loss',  # learning rate decay if validation loss reaches plateau
    factor=float(args.lr_decay_ratio),  # lr * ratio when triggered
    patience=5,
    min_lr=float(args.minimum_lr))

# Create an instance of the custom callback
mlflow_logging_callback = MLflowLoggingCallback(
    batch_size=args.batch_size,
    epochs=args.epochs,
    fine_tune=args.fine_tune,
    model_name=args.model_name,
    learning_rate=args.learning_rate,
    log_interval=1)


if __name__ == '__main__':
    if args.model_version == 0:
        # If no model version, create new model or new
        # version of an existing model.
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=float(args.learning_rate)),
            loss=VAELoss(beta=0.5, sampler=sampler),
            metrics=[RecallMetric(20, 'rec20'), RecallMetric(50, 'rec50'), NDCGMetric()])
    else:
        try:
            # Load model with pre-defined weights. A new version
            # of the model will be created, no matter if the
            # variables of currently used model have been reset
            # or not.
            path = collect_model_path(name=args.model_name+'_sampler', version=args.model_version)
            sampler = mlflow.keras.load_model(model_uri=path)

            path = collect_model_path(name=args.model_name, version=args.model_version)
            custom_objects = {'VAELoss': custom_loss_factory(sampler)}
            with custom_object_scope(custom_objects):
                model = mlflow.keras.load_model(model_uri=path)

            if args.fine_tune is False:
                # Reset model variables to start training from 0.
                # clone_model creates  a cloned network with the same
                # architecture but new model weights.
                weights = clone_model(model).get_weights()
                model.set_weights(weights)

        except mlflow.exceptions.MlflowException:
            msg = (f"The model version provided (version-{args.model_version}) doesn't exist " +
                   f"for model '{args.model_name}'.")
            raise Exception(msg)

    # Set the tracking URI programmatically
    backend_store_uri = "sqlite:///sqlite/mlflow.db"
    mlflow.set_tracking_uri(backend_store_uri)

    # Define active experiment where to save runs
    try:
        experiment_id = mlflow.create_experiment(
            name=args.experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as active_run:
        # Train the model
        history = model.fit(
            data_gen.generate_batches(),
            epochs=args.epochs,
            steps_per_epoch=num_steps_per_epoch,
            validation_data=(x_val, x_val),
            callbacks=[lr_decay, early_stop, mlflow_logging_callback])

        # Collect and log all metrics
        for metric_name in history.history:
            for i, metric_value in enumerate(history.history[metric_name]):
                mlflow.log_metric(metric_name, metric_value, step=i)

        # Log the parameters
        params = {
            "batch_size": args.batch_size,
            "epoch": args.epochs,
            "optimizer": model.optimizer._name,
            "loss": 'custom_bce_kld',
            "lr": args.learning_rate,
            "minimum_lr": args.minimum_lr,
            "regularizer": model.activity_regularizer,
            "opt_beta_1": model.optimizer.beta_1,
            "opt_beta_2": model.optimizer.beta_2,
            "fine_tune": args.fine_tune,
            "model_name": args.model_name
        }
        mlflow.log_params(params)

        # Add tags to the run
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("experiment_id", experiment_id)
        mlflow.set_tag("model_name", args.model_name)
        mlflow.set_tag("fine_tune", args.fine_tune)

        # Log artifacts when ending training process
        late_logging(
            model=model,
            sampler=sampler,
            features=x_train,
            model_name=args.model_name,
            files_path=filesdir,
            run_id=active_run.info.run_id,
            experiment_name=args.experiment_name,
            experiment_id=experiment_id
        )
