import tensorflow as tf

from tensorflow.keras.losses import Loss


class VAELoss(Loss):
    def __init__(self, beta=0.5, sampler=None, **kwargs):
        super(VAELoss, self).__init__(**kwargs)
        self.beta = beta
        self.sampler = sampler

    def call(self, x, x_pred):
        _, mu, var = self.sampler(x)

        # Calculate BCE
        bcel = tf.nn.softmax(x_pred, axis=-1)
        bcel = -tf.reduce_mean(tf.reduce_sum(tf.math.log(bcel) * x, axis=-1))

        #bce = tf.keras.losses.BinaryCrossentropy()
        #bcel = bce(x, x_pred)  # do not use softmax

        # Calculate KLD
        kld = 1 + var - tf.math.square(mu) - tf.math.exp(var)
        kld = -tf.reduce_sum(kld, axis=-1) * self.beta

        return kld + bcel


def custom_loss_factory(sampler):
    """Returns custom loss function dependent on input
    sampler object."""
    def loss_function(**kwargs):
        return VAELoss(beta=0.5, sampler=sampler, **kwargs)
    return loss_function
