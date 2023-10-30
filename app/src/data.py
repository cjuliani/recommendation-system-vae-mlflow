import numpy as np


class DataGenerator:
    """Generates data batches on-demand with shuffling mechanism to
    randomize data each time an epoch is reached, i.e. the full data
    has been generated."""
    def __init__(self, data, batch_size):
        self.data = data
        self.num_items = data.shape[1]
        self.data_size = len(data)
        self.indices = list(range(self.data_size))
        self.batch_size = batch_size

        self.shuffle_dataset()

    def shuffle_dataset(self):
        # Shuffle data based on the row indices of matrix data
        num_rows = self.data.shape[0]
        row_indices = np.random.permutation(num_rows)
        self.data = self.data[row_indices]

    def generate_batches(self):
        # Returns complete training data and shuffle it so that user comes
        # at least once in training because of multiple epochs
        while 1:
            self.shuffle_dataset()
            for i in range(0, self.data_size, self.batch_size):
                batch = self.data[i:i + self.batch_size, ...]
                yield batch, batch
