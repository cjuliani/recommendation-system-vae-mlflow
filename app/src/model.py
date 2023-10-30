import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Layer
from tensorflow.keras.models import Model


class SamplingLayer(Layer):
    def __init__(self, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, logvar, latent_dim = inputs
        if tf.keras.backend.learning_phase() == 1:
            epsilon = tf.random.normal(shape=(tf.shape(mu)[0], latent_dim), mean=0., stddev=1.)
            return mu + tf.exp(0.5 * logvar) * epsilon
        else:
            return mu


def build_model(original_dim: int, intermediate_dim: int, latent_dim: int):
    """Returns variational autoencoder model and related latent
    variable sampler.

    Args:
        original_dim: dimension of input data.
        intermediate_dim: dimension of dense layers.
        latent_dim: dimension of latent space from which
            sampling is performed.
    """
    # Define the encoder network
    x = Input(batch_shape=(None, original_dim))
    h = Dense(
        intermediate_dim,
        activation='tanh')(x)
    h = Dropout(0.5)(h)

    # Define parameters for the Gaussian distribution in the latent space
    z_mean = Dense(
        latent_dim,
        activation=None)(h)
    z_log_var = Dense(
        latent_dim,
        activation=None)(h)

    # Use a custom Lambda layer for sampling in the latent space
    z = SamplingLayer()([z_mean, z_log_var, latent_dim])

    # Define the decoder network
    h_decoder = Dense(
        intermediate_dim,
        activation='tanh')
    x_bar = Dense(
        original_dim,
        activation=None)

    # Pass the latent variable through the decoder layers
    h_decoded = h_decoder(z)
    x_decoded = x_bar(h_decoded)

    # Build and compile model
    model = Model(x, x_decoded)
    sampler = Model(x, [z, z_mean, z_log_var])

    return model, sampler


def recommend_k_items(x: np.ndarray, k: int, model, remove_seen: bool = True):
    """Returns the top-k items ordered by a relevancy score.

    Args:
        x: interaction matrix user-item.
        k: the number of items to recommend.
        model: VAE keras model.
        remove_seen: if True, rule out items used in training
            from recommended items.

    Returns:
        Two matrices, one containing the top_k elements ordered by their score,
        and the other containing related item indices.
    """
    # Load model weights and predict scores
    scores = model.predict(x, batch_size=64)

    if remove_seen:
        # If true, removes items from the train set by setting them to zero
        seen_mask = np.not_equal(x, 0)
        scores[seen_mask] = 0

    # Get the top k items and scores
    top_items = np.argpartition(-scores, range(k), axis=1)[:, :k]
    top_scores = scores[np.arange(scores.shape[0])[:, None], top_items]

    return top_items, top_scores
