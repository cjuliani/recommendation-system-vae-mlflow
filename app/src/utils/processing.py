import numpy as np


def split_data(x: np.ndarray, batch_size: int, train_ratio: float = 0.9):
    """Returns train and validation data, and the number of
    training steps per epoch."""
    data_size = x.shape[0]
    num_train = round(data_size * train_ratio)
    num_non_train = data_size - num_train
    tmp = num_non_train // 2

    # Calculate number steps per epoch given train data and
    # batch size
    if num_train % batch_size == 0:
        num_steps_per_epoch = num_train // batch_size
    else:
        num_steps_per_epoch = (num_train // batch_size) + 1

    return x[:num_train], x[num_train:num_train+tmp], x[num_train+tmp:num_train+tmp+tmp], num_steps_per_epoch


def remap_data(x):
    """Returns dataframe whose user and item elements are re-mapped
    to continuous integers starting from 0."""
    # Extract unique user and item IDs
    unique_users = set(x['user'])
    unique_items = set(x['item'])

    # Create a mapping dictionaries
    user_id_to_index = {user_id: index for index, user_id in enumerate(unique_users)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(unique_items)}

    # Re-map values of user and item ids so that hey start from 0
    x['user'] = x['user'].map(lambda x: user_id_to_index[x])
    x['item'] = x['item'].map(lambda x: item_id_to_index[x])

    return x


def create_interaction_matrix(train_data, min_items_inter=60, min_user_inter=3):
    """Returns binary matrix of user-item interactions.

    Args:
        train_data: dataframe containing the user and item IDs.
        min_items_inter: minimum number of user interactions per
            individual item.
        min_user_inter: minimum number of interactions with items
            per user.
    """
    # Extract unique user and item IDs
    unique_users = train_data['user'].unique()
    unique_items = train_data['item'].unique()

    # Create a mapping from non-continuous IDs to continuous indices
    user_id_to_index = {user_id: index for index, user_id in enumerate(unique_users)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(unique_items)}

    # Create an empty LIL matrix with the specified shape
    matrix = np.zeros((len(unique_users), len(unique_items)), dtype=np.float32)

    # Iterate over the ratings and populate the adjacency matrix
    for index, data in train_data.iterrows():
        user_index = user_id_to_index[data['user']]
        item_index = item_id_to_index[data['item']]
        matrix[user_index, item_index] = 1.  # user-item interactions only (binary)
        # matrix[user_index, item_index] = rating['label']  # user-item interactions with rating consideration

    # Keep items whose number of interactions exceeds given minimum
    indices = np.where(np.sum(matrix, axis=0) >= min_items_inter)[0]
    matrix = matrix[:, indices]

    # Keep users whose number of interactions exceeds given minimum
    indices = np.where(np.sum(matrix, axis=-1) >= min_user_inter)[0]

    return matrix[indices]
