import mlflow
import tensorflow as tf


# Define a custom Keras callback for logging metrics and parameters
class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, epochs, fine_tune, model_name, learning_rate, log_interval):
        super().__init__()
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune = fine_tune
        self.model_name = model_name
        self.learning_rate = learning_rate

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            # Log metrics
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)

            # Log parameters (you can customize this part)
            params = {
                "batch_size": self.batch_size,
                "epoch": self.epochs,
                "optimizer": self.model.optimizer._name,
                "loss": 'custom_bce_kld',
                "lr": self.learning_rate,
                "fine_tune": self.fine_tune,
                "model_name": self.model_name
            }
            mlflow.log_params(params)
