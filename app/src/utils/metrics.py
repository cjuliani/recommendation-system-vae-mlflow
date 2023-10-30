import tensorflow as tf


class NDCGMetric(tf.keras.metrics.Metric):
    """Calculates Normalized Discounted Cumulative Gain (NDCG) metric
     at top k elements for binary relevance.

    All the 0's in binary data indicate 0 relevance measures the quality
    of a recommendation list, particularly how well it ranks relevant items.
    NDCG takes into account both the relevance of the recommended items and
    their positions in the list. The relevance of items is usually assessed
    based on user interactions (e.g., clicks, ratings) or ground truth data.
    The metric discounts the gain of items that are ranked lower in the list.

    The higher the NDCG scores, the better the model is at providing relevant
    recommendations to users.
    """
    def __init__(self, k=100, name='ndcg', **kwargs):
        super(NDCGMetric, self).__init__(name=name, **kwargs)
        self.k = k
        self.score = self.add_weight('top_k', initializer='zeros')
        self.discount = 1. / tf.math.log(tf.range(2, k + 2, dtype=tf.float32))

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get the top-k recommendations for each user
        top_k_indices = tf.math.top_k(y_pred, k=self.k).indices
        batch_size = tf.shape(y_true)[0]

        # Calculate NDCG for each user (batch element)
        ndcg = 0.
        for i in tf.range(batch_size):
            idx_topk = tf.gather(top_k_indices[i], tf.argsort(-tf.gather(y_pred[i], top_k_indices[i])))
            DCG = tf.reduce_sum(tf.gather(y_true[i], idx_topk) * self.discount)
            IDCG = tf.reduce_sum(tf.sort(y_true[i], direction='DESCENDING')[:self.k] * self.discount)
            ndcg += DCG / IDCG

        self.score.assign_add(ndcg)

    def result(self):
        return self.score / tf.cast(self.k, dtype=tf.float32)

    def reset_state(self):
        self.score.assign(0)


class RecallMetric(tf.keras.metrics.Metric):
    """ Calculates the recall metric measuring the ability of a recommendation
    system to retrieve all relevant items. It is typically used to evaluate the
    ability of a system to find relevant items from a larger pool of possibilities.
    Recall is defined as the ratio of relevant items retrieved by the system to the
    total number of relevant items.

    The higher the recall scores, the better the model is at providing relevant
    recommendations to users.
    """
    def __init__(self, k=100, name='recall', **kwargs):
        super(RecallMetric, self).__init__(name=name, **kwargs)
        self.k = k
        self.true_positives = self.add_weight('true_positives', initializer='zeros')
        self.total_positives = self.add_weight('total_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get the top-k recommendations for each user
        top_k_indices = tf.math.top_k(y_pred, k=self.k).indices
        batch_size = tf.shape(y_true)[0]

        # Calculate recall for each user (batch element)
        for i in range(batch_size):
            true_positives = tf.reduce_sum(tf.gather(y_true[i], top_k_indices[i]))
            total_positives = tf.reduce_sum(y_true[i])
            self.true_positives.assign_add(true_positives)
            self.total_positives.assign_add(total_positives)

    def result(self):
        recall = self.true_positives / (self.total_positives + tf.keras.backend.epsilon())
        return recall

    def reset_state(self):
        self.true_positives.assign(0)
        self.total_positives.assign(0)
