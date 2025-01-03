from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

from utils import *


def create_model(layers=2, n_units=20, init_mode='glorot_normal', activation='tanh', lmb=0.0001, eta=0.001, alpha=0.7):
    """
    Create a Keras Sequential model with the specified hyperparameters.
    """
    model = Sequential()

    # Add hidden layers
    for _ in range(layers):
        model.add(Dense(n_units, kernel_initializer=init_mode, activation=activation, kernel_regularizer=l2(lmb)))

    # Add output layer
    model.add(Dense(3, activation='linear', kernel_initializer=init_mode))

    # Compile the model with SGD optimizer
    optimizer = SGD(learning_rate=eta, momentum=alpha)
    model.compile(optimizer=optimizer, loss=euclidean_distance_loss)

    return model


def find_best_params(x, y, epochs=200, n_splits=5):
    """
    Perform grid search for hyperparameter optimization.
    """

    # Fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # Parameter grid
    param_grid = {
        "lmb": [0.0001, 0.001],
        "eta": [0.001, 0.01],
        "alpha": [0.7, 0.9],
        "batch_size": [32, 64]
    }

    # Initialize tracking variables
    best_score = float('inf')
    best_params = None

    # K-Fold setup
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Start grid search
    print("Starting Grid Search...\n")
    start_time = time.time()

    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        fold_losses = []

        for train_idx, val_idx in kfold.split(x):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create model with current parameters
            model = create_model(
                lmb=params['lmb'],
                eta=params['eta'],
                alpha=params['alpha']
            )

            # Train the model
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=params['batch_size'],
                verbose=0
            )

            # Record validation loss
            fold_losses.append(history.history['val_loss'][-1])

        # Average validation loss across folds
        avg_val_loss = np.mean(fold_losses)
        print(f"Average validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            best_params = params

    print("\nGrid Search completed in {:.4f} seconds.\n".format(time.time() - start_time))
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_score:.4f}\n")

    best_params['epochs'] = epochs

    return best_params


def predict(model, x_ts, x_its, y_its):
    """
    Predict using the trained model and calculate loss for internal test set.
    """

    y_ipred = model.predict(x_its)
    iloss = euclidean_distance_loss(y_its, y_ipred)

    y_pred = model.predict(x_ts)

    return y_pred, K.eval(iloss)


def plot_learning_curve(history, start_epoch=1, savefig=False, **kwargs):
    """
    Plot the learning curve for the Keras model training process.
    """
    legend = ['Loss TR']
    plt.plot(range(start_epoch, kwargs['epochs']), history['loss'][start_epoch:])
    if "val_loss" in history:
        plt.plot(range(start_epoch, kwargs['epochs']), history['val_loss'][start_epoch:])
        legend.append('Loss VL')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'Keras Learning Curve \n {kwargs}')
    plt.legend(legend)

    if savefig:
        save_figure("kerasNN", **kwargs)

    plt.show()


def main(ms=False):
    """
    Main function to execute the Keras Neural Network pipeline.
    """
    print("Keras pipeline started")

    # Read training data
    x, x_its, y, y_its = read_tr(split=0.2)

    # Perform model selection or use default parameters
    if ms:
        params = find_best_params(x, y)
    else:
        params = dict(eta=0.002, alpha=0.7, lmb=0.0001, epochs=200, batch_size=64)

    # Build and train the model
    model = create_model(eta=params['eta'], alpha=params['alpha'], lmb=params['lmb'])
    res = model.fit(x, y, validation_split=0.3, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1)

    # Training and validation losses
    tr_losses = res.history['loss']
    val_losses = res.history['val_loss']

    y_pred, ts_losses = predict(model=model, x_ts=read_ts(), x_its=x_its, y_its=y_its)

    print("TR Loss: ", tr_losses[-1])
    print("VL Loss: ", val_losses[-1])
    print("TS Loss: ", np.mean(ts_losses))

    # Plot learning curve
    plot_learning_curve(res.history, savefig=True, **params)

    print("Keras pipeline ended")


if __name__ == '__main__':
    main(ms=True)
