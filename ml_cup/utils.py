import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

# Suppress TensorFlow INFO, WARNING, and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_tr(split=None):
    """
    Read the training set.
    
    Args:
        split (bool): If True, split the dataset into development and internal test set.
    
    Returns:
        - If split is False: (x, y)
        - If split is True: (x_train, y_train, x_test, y_test)
    """
    file = os.path.join(ROOT_DIR, "dataset", "ML-CUP24-TR.csv")
    train = loadtxt(file, delimiter=',', usecols=range(1, 16), dtype=np.float32)

    x = train[:, :-3]
    y = train[:, -3:]

    if split:
        return train_test_split(x, y, test_size=split, random_state=42)
    else:
        return x, y


def read_ts():
    """
    Read the blind test set.
    
    Returns:
        np.ndarray: The blind test set.
    """
    file = os.path.join(ROOT_DIR, "dataset", "ML-CUP24-TS.csv")
    test = loadtxt(file, delimiter=',', usecols=range(1, 13), dtype=np.float64)

    return test


def save_figure(model_name, **params):
    """
    Save a plot as a PNG image file.
    
    Args:
        model_name (str): The name of the model.
        **params: Additional parameters to include in the file name.
    """
    # Construct file name
    param_str = "_".join(f"{k}{v}" for k, v in params.items())
    file_name = f"{param_str}.png"

    # Create directory if it doesn't exist
    dir_path = os.path.join(ROOT_DIR, model_name, "plot")
    os.makedirs(dir_path, exist_ok=True)

    # Save the plot
    fig_path = os.path.join(dir_path, file_name)
    plt.savefig(fig_path, dpi=600)


def write_blind_results(model_name, y_pred):
    """
    Save predicted results in a CSV file for the blind test dataset.
    
    Args:
        y_pred (np.ndarray): The predictions to save.
    """

    assert len(y_pred) == 500, "Not enough predictions! 500 predictions expected!"

    file_path = os.path.join(ROOT_DIR, model_name,"P&R_ML-CUP24-TS.csv")
    with open(file_path, "w") as f:
        f.write("# Giuseppe Di Palma \t Daniel Wahle\n")
        f.write("# P&R\n")
        f.write("# ML-CUP24 v1\n")
        f.write("# 22/01/2025\n")

        for pred_id, p in enumerate(y_pred, start=1):
            f.write(f"{pred_id},{p[0]},{p[1]},{p[2]}\n")

    f.close()


def euclidean_distance_loss(y_true, y_pred):
    """
    Compute the Euclidean distance loss for Keras models.
    
    Args:
        y_true (tensor): Ground truth values.
        y_pred (tensor): Predicted values.
    
    Returns:
        tensor: The Euclidean distance loss.
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def euclidean_distance_score(y_true, y_pred):
    """
    Compute the mean Euclidean distance between predictions and true values.
    
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
    
    Returns:
        float: The mean Euclidean distance.
    """
    return np.mean(euclidean_distance_loss(y_true, y_pred))


# Define a custom scorer for sklearn
scorer = make_scorer(euclidean_distance_score, greater_is_better=False)
