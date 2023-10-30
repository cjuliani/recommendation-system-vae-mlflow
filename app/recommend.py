import mlflow.keras
import numpy as np
import pickle
import argparse

from src.model import recommend_k_items
from src.utils.errors import handle_bool_parsing
from src.utils.utils import strtobool

from tensorflow.keras.utils import custom_object_scope
from src.utils.loss import custom_loss_factory
from src.utils.utils import collect_model_path


if __name__ == '__main__':
    bool_checker = lambda x: bool(strtobool(handle_bool_parsing(x, '--fine_tune')))
    parser = argparse.ArgumentParser(description="Evaluate a machine learning model")
    parser.add_argument("--model_name", type=str, default='Unknown', help="Name of existing model to load.")
    parser.add_argument("--model_version", type=int, default=0, help="Version of model loaded.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of items to recommend.")
    parser.add_argument(
        "--remove_train_items", default=False, type=bool_checker, help="If True, train items not recommended.")
    args = parser.parse_args()

    # Load evaluation data
    artifact_path = collect_model_path(name=args.model_name, version=args.model_version)
    artifact_path = artifact_path.split('///')[-1]
    with open(artifact_path + '/files/' + 'x_train.pkl', 'rb') as file:
        x_train = pickle.load(file)

    original_dim = x_train.shape[1]  # number of items

    # Load latent variable sampler and VAE model
    path = collect_model_path(name=args.model_name + '_sampler', version=args.model_version)
    sampler = mlflow.keras.load_model(model_uri=path)

    path = collect_model_path(name=args.model_name, version=args.model_version)
    custom_objects = {'VAELoss': custom_loss_factory(sampler)}
    with custom_object_scope(custom_objects):
        model = mlflow.keras.load_model(model_uri=path)

    # Calculate number steps given batch size
    if x_train.shape[1] % args.batch_size == 0:
        num_steps_per_epoch = x_train.shape[1] // args.batch_size
    else:
        num_steps_per_epoch = (x_train.shape[1] // args.batch_size) + 1

    # Process data for recommendation. This is an example of
    # code for demonstrative purpose. The average score calculated
    # has no meaning.
    recommendations = []
    for i in range(num_steps_per_epoch):
        start = i * args.batch_size
        end = start + args.batch_size

        top_items, top_scores = recommend_k_items(
            x=x_train[start:end],
            k=args.top_k,
            model=model,
            remove_seen=args.remove_train_items)
        recommendations.append(top_scores)

    print('average score:', np.mean(np.concatenate(recommendations)))
