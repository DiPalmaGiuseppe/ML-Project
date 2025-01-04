import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from monk_dataset import monk_loader
from tabulate import tabulate
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import numpy as np
import os

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DIR_PATH = "./monk_tensorflow_fig"

# 1. Funzione per preparare i dati
def prepare_data(dataset_idx):
    X_train, y_train, X_test, y_test = monk_loader(dataset_idx)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 2. Funzione per creare il modello MLP
def build_model(n_units, learning_rate, momentum, lmb):
    model = models.Sequential([
        layers.Dense(n_units, activation='tanh', input_dim=17, kernel_regularizer = lmb),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[BinaryAccuracy(name='accuracy')])
    return model

def save_monk_fig(monk_number, eta, alpha, plt_type='loss'):

    name = "monk"+monk_number+f"_{plt_type}_eta{eta}_alpha{alpha}.png"

    # create plot directory if it doesn't exist
    os.makedirs(DIR_PATH, exist_ok=True)

    # save plot as figure
    fig_path = os.path.join(DIR_PATH, name)
    plt.savefig(fig_path, dpi=600)
    plt.clf()

# 3. Funzione per effettuare la GridSearch
def grid_search(idx, reg = False):
    print(f"---- Dataset {idx} ----")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(idx)
    
    param_grid = {
        'n_units': [2, 3, 4],
        'momentum': [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],
        'batch_size': [16, 32]
    }

    if reg:
        param_grid["lmb"] = [0.0005, 0.001]
    grid = ParameterGrid(param_grid)
    best_res = None
    best_params = None
    best_score = -np.inf
    best_model = None

    for params in grid:
        if reg:
            lmb = l2(params['lmb'])
        else:
            lmb = None
        model = build_model(
            n_units=params['n_units'],
            learning_rate=params['learning_rate'],
            momentum=params['momentum'],
            lmb = lmb
        )

        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            min_delta=0.001
        )

        res = model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=params['batch_size'],
            verbose=0,
            validation_data=(X_val, y_val), 
            callbacks=[early_stopping]
        )

        val_accuracy = res.history['val_accuracy'][-1]

        print(params)
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_params = params
            best_model = model
            best_res = res
            
            
    str_idx = str(idx)+" (reg)" if reg else str(idx)
    
    plt.plot(best_res.history['loss'])
    plt.plot(best_res.history['val_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(['Loss TR', 'Loss VL'], loc='center right')
    plt.title(f"""MONK-{str_idx} 
              (eta={best_params['learning_rate']},
              alpha={best_params['momentum']},
              batch_size={best_params['batch_size']},
              n_units={best_params['n_units']})""" +
              (f"lmb= {best_params['lmb']}- Loss" if reg else"- Loss"))
    save_monk_fig(str_idx, best_params["learning_rate"], best_params["momentum"])

    # plot results for "test" (validation) set
    plt.plot(best_res.history['accuracy'])
    plt.plot(best_res.history['val_accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['Accuracy TR', 'Accuracy TS'], loc='center right')
    plt.title(f"""MONK-{str_idx} 
              (eta={best_params['learning_rate']},
              alpha={best_params['momentum']},
              batch_size={best_params['batch_size']},
              n_units={best_params['n_units']})""" +
              (f"lmb= {best_params['lmb']}- Accuracy" if reg else"- Accuracy"))
    save_monk_fig(str_idx, best_params["learning_rate"], best_params["momentum"], plt_type = 'acc')

    # Valutazione sui dati di test con il modello migliore
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return {
        "Dataset": str_idx,
        "Best_Params": best_params,
        "Test_Accuracy": accuracy,
        "Test_F1": f1,
        "Test_AUC": auc,
        "Training Loss": min(best_res.history['loss']),
        "Validation Loss": min(best_res.history['val_loss']),
        "Training Accuracy": max(best_res.history['accuracy']),
        "Validation Accuracy": max(best_res.history['val_accuracy']),
        "Val_Accuracy": best_score
    }

if __name__ == "__main__":
    results = []
    for i in range(1, 4):
        result = grid_search(i)
        results.append(result)
    
    result = grid_search(3, True)
    results.append(result)
   
    print(tabulate(results, headers="keys", tablefmt="pretty"))
