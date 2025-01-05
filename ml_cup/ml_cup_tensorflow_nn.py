from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

from utils import *


def create_model(layers, n_units, learning_rate, momentum, activation = 'tanh', init_mode='glorot_normal'):
    """
    Create a Keras Sequential model with the specified hyperparameters.
    """
    model = Sequential()

    model.add(Dense(n_units, kernel_initializer=init_mode, activation=activation, input_dim = 12))

    # Add hidden layers
    for _ in range(layers):
        model.add(Dense(n_units, kernel_initializer=init_mode, activation=activation))

    # Add output layer
    model.add(Dense(3, activation='linear', kernel_initializer=init_mode))

    # Compile the model with SGD optimizer
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss=euclidean_distance_loss)

    return model


def evaluate_params(params, x, y, n_splits, seed, epochs):
    
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import EarlyStopping
    # K-Fold setup
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_losses = []

    for train_idx, val_idx in kfold.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create model with current parameters
        model = create_model(params['n_hidden_layers'], params['n_units'], params['learning_rate'], params['momentum'])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping]
        )

        fold_losses.append(min(history.history['val_loss']))

    avg_val_loss = np.mean(fold_losses)
    print(f"Parameters {params} - avg val loss {avg_val_loss}")    
    return params, avg_val_loss

def find_best_params(x, y, epochs=500, n_splits=10):
    """
    Perform grid search for hyperparameter optimization with parallelization.
    """
    # Fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    param_grid = {
        'n_hidden_layers': [0, 1, 2],
        'n_units': [5, 10, 15, 20, 30],
        'momentum': [0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    }

    # Flatten the parameter grid
    param_list = ParameterGrid(param_grid)

    # Start parallel processing
    print("Starting Grid Search with Parallelization...\n")

    start_time = time.time()
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            evaluate_params, 
            [(params, x, y, n_splits, seed, epochs) for params in param_list]
        )

    # Find the best parameters
    best_score = float('inf')
    best_params = None

    for params, avg_val_loss in results:
        print(f"Tested parameters: {params} - Avg val loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_score:
            best_score = avg_val_loss
            best_params = params

    print("\nGrid Search completed in {:.4f} seconds.\n".format(time.time() - start_time))
    # print(f"Best validation loss: {best_score:.4f}\n")

    best_params['epochs'] = epochs

    print("Best validation loss:", best_score)

    return best_params

def main(ms=False):
    """
    Main function to execute the Keras Neural Network pipeline.
    """
    print("Keras pipeline started")

    # Read training data
    X_train, X_test, y_train, y_test = read_tr(split = 0.2)
    X_blind = read_ts()
    
    # Scale the data
    X_train, X_test, X_blind, y_train, feature_scaler, target_scaler = scale_data(
        X_train, X_test, X_blind, y_train
    )

    # Perform model selection or use default parameters
    if ms:
        params = find_best_params(X_train, y_train)
    else:
        params = dict(learning_rate=0.002, momentum=0.7, lmb=0.0001, epochs=500, batch_size=32)

    # Build and train the model
    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    model = create_model(params['n_hidden_layers'], params['n_units'], params['learning_rate'], params['momentum'])
    res = model.fit(X_train, y_train, validation_split=0.2, epochs=params['epochs'], batch_size=32, verbose=1,callbacks=[early_stopping])

    # Training and validation losses
    tr_losses = res.history['loss']
    val_losses = res.history['val_loss']
    
    plt.plot(tr_losses, label='Loss TR')
    plt.plot(val_losses, label='Loss VL')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"""(
        eta={params['learning_rate']}, 
        alpha={params['momentum']},
        layers={params['n_hidden_layers']},
        n_units={params['n_units']}
        ) - Loss""")
    save_figure("KerasNN",
                eta = params['learning_rate'],
                alpha = params['momentum'],
                layers = params['n_hidden_layers'],
                n_units = params['n_units'],
                plt_type='loss')
    
    y_pred = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred)
    ts_loss = euclidean_distance_score(y_pred, y_test)
    r2 = r2_score(y_pred, y_test)

    print(f"Best config {params}")
    print("TS Loss:", ts_loss)
    print("Test R²:", r2)
    # print("TR Loss:", min(tr_losses))
    # print("VL Loss:", min(val_losses))
    
    y_pred = model.predict(X_blind)
    
    y_pred = target_scaler.inverse_transform(y_pred)
    write_blind_results("KerasNN", y_pred)

    print("Keras pipeline ended")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main(ms=True)
