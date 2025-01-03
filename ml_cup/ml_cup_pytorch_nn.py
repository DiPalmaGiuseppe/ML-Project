import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from utils import read_tr, read_ts, write_blind_results, save_figure

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_hidden_layer):
        super(MLPModel, self).__init__()
        self.activation_function = nn.LeakyReLU
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_size))
        layers.append(self.activation_function())
        
        for _ in range(num_hidden_layer):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(self.activation_function())
        
        layers.append(nn.Linear(hidden_layer_size, 3))
        layers.append(self.activation_function())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)
    
def mean_euclidean_error(y_real, y_pred):
    return torch.mean(F.pairwise_distance(y_real, y_pred, p=2))

def train_and_evaluate(model, optimizer, train_dataloader, val_dataloader, patience=10):
    criterion = mean_euclidean_error
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(200):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)

        train_losses.append(train_loss / len(train_dataloader.dataset))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * len(x_batch)

        val_loss /= len(val_dataloader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_val_loss != float('inf'):
        model.load_state_dict(best_model_state)
    return train_losses, val_losses

def train_for_config(config, X_train, y_train, kfold, train_dataset):
    fold_val_losses = []

    for train_idx, val_idx in kfold.split(X_train):
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False)

        model = MLPModel(
            input_size=X_train.shape[1],
            num_hidden_layer=config['n_hidden_layers'],
            hidden_layer_size=config['n_units'],
        )
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

        _, val_losses = train_and_evaluate(model, optimizer, train_dataloader, val_dataloader)

        fold_val_losses.append(min(val_losses))

    avg_val_loss = np.mean(fold_val_losses)
    return config, avg_val_loss

# Funzione per eseguire la grid search con cross validation
def grid_search():
    print("---- Grid Search ----")
    param_grid = {
        'n_hidden_layers': [0, 1, 2],
        'n_units': [5, 10, 15, 20, 30],
        'momentum': [0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    }

    X_train, X_test, y_train, y_test = read_tr(split = 0.2)
    X_Blind = read_ts()
    
    X_train = torch.tensor(X_train, dtype = torch.float32)
    X_test = torch.tensor(X_test, dtype = torch.float32)
    X_Blind = torch.tensor(X_Blind, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    best_config = None
    best_val_loss = float('inf')

    grid = ParameterGrid(param_grid)

    # Risultati della grid search
    grid_search_results = {}

    with ProcessPoolExecutor() as executor:
        futures = []
        for config in grid:
            futures.append(executor.submit(train_for_config, config, X_train, y_train, kfold, train_dataset))

        for future in futures:
            res_config, res_avg_loss = future.result()
            grid_search_results[tuple(res_config.items())] = res_avg_loss
            print(res_config, res_avg_loss)

    best_config = min(grid_search_results, key=grid_search_results.get)
    best_val_loss = grid_search_results[best_config]

    best_config = dict(best_config) 
    print("Best configuration:", best_config)
    print("Best validation loss:", best_val_loss)

    # Creazione del modello finale con la miglior configurazione
    best_model = MLPModel(
            input_size=X_train.shape[1],
            num_hidden_layer=best_config['n_hidden_layers'],
            hidden_layer_size=best_config['n_units']
        )
    optimizer = optim.SGD(best_model.parameters(), lr=best_config['learning_rate'], momentum=best_config['momentum'])
    
    # Splitting del dataset per il training finale
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

    train_losses, val_losses = train_and_evaluate(best_model, optimizer, train_dataloader, val_dataloader, 20)

    best_model.eval()
    y_blind_pred = None
    with torch.no_grad():
        y_blind_pred = best_model(X_Blind)
    write_blind_results("TorchNN", y_blind_pred)

    # Plot del grafico delle perdite
    plt.plot(train_losses, label='Loss TR')
    plt.plot(val_losses, label='Loss CV')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"(eta={best_config['learning_rate']}, alpha={best_config['momentum']}) - Loss")
    save_figure("TorchNN", eta = best_config['learning_rate'], alpha = best_config['momentum'], plt_type='loss')

    print("---- Final Results ----")
    y_pred = None
    with torch.no_grad():
        y_pred = best_model(X_test)
    test_loss = mean_euclidean_error(y_test, y_pred)
    print("Test Loss:", test_loss.item())
    ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
    ss_res = torch.sum((y_test - y_pred) ** 2)
    r2_score = 1 - ss_res / ss_total
    print("Test R²:", r2_score.item())

if __name__ == "__main__":
    grid_search()
