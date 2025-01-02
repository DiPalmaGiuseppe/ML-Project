import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from monk_dataset import monk_loader
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory per salvare i grafici
DIR_PATH = "./monk_pytorch_fig"
os.makedirs(DIR_PATH, exist_ok=True)

# 1. Funzione per la preparazione dei dati
def prepare_data(dataset_idx):
    X_train, y_train, X_test, y_test = monk_loader(dataset_idx)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    return X_train, y_train, X_test, y_test

# 2. Creazione del modello MLP
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

# Funzione per salvare i grafici
def save_monk_fig(monk_number, eta, alpha, plt_type='loss'):
    name = f"monk{monk_number}_{plt_type}_eta{eta}_alpha{alpha}.png"
    fig_path = os.path.join(DIR_PATH, name)
    plt.savefig(fig_path, dpi=600)
    plt.clf()

# 3. Funzione per addestrare e valutare il modello
def train_and_evaluate(model, optimizer, train_dataloader, val_dataloader, patience=10):
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(200):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
            train_correct += ((y_pred > 0.5).int() == y_batch.int()).sum().item()
            train_total += len(x_batch)

        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * len(x_batch)
                val_correct += ((y_pred > 0.5).int() == y_batch.int()).sum().item()
                val_total += len(x_batch)

        val_loss /= val_total
        val_losses.append(val_loss)
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_model_state)
    return train_losses, val_losses, train_accuracies, val_accuracies

# 4. Funzione per eseguire la grid search
def grid_search():
    print("---- Grid Search ----")
    param_grid = {
        'n_units': [2, 3, 4],
        'momentum': [0.7, 0.75, 0.8, 0.85, 0.9],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    results = []

    for dataset_idx in range(1, 4):  # Itera su MONK-1, MONK-2, MONK-3
        print(f"Dataset: MONK-{dataset_idx}")
        X_train, y_train, X_test, y_test = prepare_data(dataset_idx)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        best_config = None
        best_val_acc = 0
        best_model = None
        best_train_losses = None
        best_val_losses = None
        best_train_accuracies = None    
        best_val_accuracies = None

        grid = ParameterGrid(param_grid)

        for config in grid:
            print(f"Testing config: {config}")

            model = MLPModel(input_size=X_train.shape[1], hidden_layer_size=config['n_units'])
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

            train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
                model, optimizer, train_dataloader, val_dataloader
            )

            if max(val_accuracies) > best_val_acc:
                best_val_acc = max(val_accuracies)
                best_config = config
                best_model = model
                best_train_losses = train_losses
                best_val_losses = val_losses
                best_train_accuracies = train_accuracies
                best_val_accuracies = val_accuracies

        # Salva i grafici
        plt.plot(best_train_losses, label='Loss TR')
        plt.plot(best_val_losses, label='Loss VL')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"MONK-{dataset_idx} (eta={best_config['learning_rate']}, alpha={best_config['momentum']}) - Loss")
        save_monk_fig(dataset_idx, best_config['learning_rate'], best_config['momentum'], plt_type='loss')

        plt.plot(best_train_accuracies, label='Accuracy TR')
        plt.plot(best_val_accuracies, label='Accuracy VL')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"MONK-{dataset_idx} (eta={best_config['learning_rate']}, alpha={best_config['momentum']}) - Accuracy")
        save_monk_fig(dataset_idx, best_config['learning_rate'], best_config['momentum'], plt_type='acc')

        # Valutazione sui dati di test
        best_model.eval()
        y_pred = best_model(X_test).detach().cpu().numpy().round()
        accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
        f1 = f1_score(y_test.cpu().numpy(), y_pred)
        auc = roc_auc_score(y_test.cpu().numpy(), y_pred)

        results.append({
            "Dataset": dataset_idx,
            "Test_Accuracy": accuracy,
            "Test_F1": f1,
            "Test_AUC": auc,
            "Best_Config": best_config,
        })

    # Salva i risultati in un file
    with open(DIR_PATH+"/results.txt", "w") as f:
        f.write("---- Final Results ----\n")
        for result in results:
            f.write(f"\nDataset MONK-{result['Dataset']}:\n")
            f.write(f"Test Accuracy: {result['Test_Accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {result['Test_F1']:.4f}\n")
            f.write(f"Test AUC: {result['Test_AUC']:.4f}\n")
            f.write(f"Best Config: {result['Best_Config']}\n")

    print("\n---- Final Results ----")
    for result in results:
        print(result)

if __name__ == "__main__":
    grid_search()
