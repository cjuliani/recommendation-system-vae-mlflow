import mlflow.keras
import argparse
import pickle

from tensorflow.keras.utils import custom_object_scope
from src.utils.loss import custom_loss_factory
from src.utils.utils import collect_model_path
from src.utils.utils import create_or_set_path, save_histogram


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a machine learning model")
    parser.add_argument("--model_name", type=str, default='Unknown', help="Name of existing model to load.")
    parser.add_argument("--model_version", type=int, default=0, help="Version of model loaded.")
    parser.add_argument("--batch_size", type=int, default=25, help="Training batch size.")
    args = parser.parse_args()

    # Load evaluation data
    artifact_path = collect_model_path(name=args.model_name, version=args.model_version)
    artifact_path = artifact_path.split('///')[-1]
    with open(artifact_path + '/files/' + 'x_eval.pkl', 'rb') as file:
        x_eval = pickle.load(file)

    original_dim = x_eval.shape[1]  # number of items

    # Load latent variable sampler and VAE model
    path = collect_model_path(name=args.model_name + '_sampler', version=args.model_version)
    sampler = mlflow.keras.load_model(model_uri=path)

    path = collect_model_path(name=args.model_name, version=args.model_version)
    custom_objects = {'VAELoss': custom_loss_factory(sampler)}
    with custom_object_scope(custom_objects):
        model = mlflow.keras.load_model(model_uri=path)

    # Evaluate the model
    results = model.evaluate(x_eval, x_eval, batch_size=args.batch_size)

    # Save histogram of evaluation scores
    metrics_data = {}
    for i, metric in enumerate(model.metrics):
        metrics_data[metric.name] = results[i]

    base_path = create_or_set_path(artifact_path + '/info_eval/')
    save_histogram(
        metrics_data=metrics_data,
        save_path=base_path + "evaluation_metrics_histogram.png"
    )
