import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tabulate import tabulate
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from monk_dataset import monk_loader
import torchmetrics


# 1. Funzione per la preparazione dei dati
def prepare_data(dataset_idx):
    X_train, y_train, X_test, y_test = monk_loader(dataset_idx)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    return X_train, y_train, X_test, y_test


# 2. Creazione del modello MLP
class MLPModel(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, activation_fn, learning_rate, momentum=0.9):
        super(MLPModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_size))
        layers.append(activation_fn())
        layers.append(nn.Linear(hidden_layer_size, 1))  # Output layer
        layers.append(nn.Sigmoid())  # Sigmoid per classificazione binaria
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        return self.network(x).squeeze(1)  # Assicura che l'output sia della forma [batch_size]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.BCELoss()(y_pred, y)
        acc = self.train_accuracy(y_pred, y.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        acc = self.val_accuracy(y_pred, y.int())
        return acc

    def on_validation_epoch_end(self):
        avg_val_acc = self.val_accuracy.compute()
        self.log("val_acc", avg_val_acc)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer


# 3. Creazione del modello di regressione logistica
class LogisticRegressionModel(pl.LightningModule):
    def __init__(self, input_size, learning_rate):
        super(LogisticRegressionModel, self).__init__()
        self.model = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        return self.sigmoid(self.model(x)).squeeze(1)  # Assicura la forma corretta dell'output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.BCELoss()(y_pred, y)
        acc = self.train_accuracy(y_pred, y.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        acc = self.train_accuracy(y_pred, y.int())
        return acc

    def on_validation_epoch_end(self):
        avg_val_acc = self.train_accuracy.compute()
        self.log("val_acc", avg_val_acc)
        self.train_accuracy.reset()

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)


# 4. Funzione per allenare un modello
def train_model(model, train_dataloader, val_dataloader):
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[
            EarlyStopping(monitor="val_acc", patience=5, mode="max"),
            ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")
        ],
        logger=False,
        enable_model_summary=False
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return trainer


# 5. Funzione per la ricerca esaustiva degli iperparametri
def exhaustive_search():
    print("---- Exhaustive Search with K-Fold ----")
    input_size = monk_loader(1)[0].shape[1]

    # Iperparametri da esplorare
    hidden_layer_sizes_options = [2, 3, 4]
    activation_options = [nn.ReLU, nn.Tanh, nn.Sigmoid]
    learning_rate_options = [0.001, 0.01, 0.1]
    momentum_options = [0.7, 0.8, 0.9, 0.95, 0.99]
    k_folds = 5

    # Lista per raccogliere i risultati
    results = []

    for i in range(3):
        # 6. Prepara i dati per il dataset
        X_train, y_train, X_test, y_test = prepare_data(i + 1)
        dataset = TensorDataset(X_train, y_train)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 7. Ricerca esaustiva per MLP
        best_model, best_val_acc, best_params = None, 0, {}
        for hidden_layer_sizes, activation_fn, learning_rate, momentum in product(hidden_layer_sizes_options, activation_options, learning_rate_options, momentum_options):
            val_accs = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                train_dataloader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=32, shuffle=True)
                val_dataloader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=32, shuffle=False)

                model = MLPModel(input_size, hidden_layer_sizes, activation_fn, learning_rate, momentum)
                trainer = train_model(model, train_dataloader, val_dataloader)

                val_acc = trainer.callback_metrics["val_acc"].item()
                val_accs.append(val_acc)

            avg_val_acc = np.mean(val_accs)
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model = model
                best_params = {"hidden_layer_sizes": hidden_layer_sizes, "activation_fn": activation_fn.__name__, "learning_rate": learning_rate, "momentum": momentum}

        # 8. Predizione e valutazione per MLP
        y_pred = best_model(X_test).detach().cpu().numpy().round()
        accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
        f1 = f1_score(y_test.cpu().numpy(), y_pred)
        auc = roc_auc_score(y_test.cpu().numpy(), y_pred)
        results.append({
            "Model": "MLP",
            "Dataset": i + 1,
            "Best_Params": best_params,
            "Test_Accuracy": accuracy,
            "Test_F1": f1,
            "Test_AUC": auc,
            "Val_Accuracy": best_val_acc
        })

        # 9. Ricerca esaustiva per la regressione logistica
        best_logreg_model, best_logreg_val_acc, best_logreg_param = None, 0, {}
        penalty_options = ['l2', 'l1', None]
        optimizer_options = [optim.SGD, optim.Adam, optim.RMSprop]
        weight_decay_options = [0.0001, 0.001, 0.01]

        for penalty, optimizer_fn, weight_decay, learning_rate in product(penalty_options, optimizer_options, weight_decay_options, learning_rate_options):
            val_accs = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                train_dataloader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=32, shuffle=True)
                val_dataloader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=32, shuffle=False)

                model = LogisticRegressionModel(input_size, learning_rate)
                optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                trainer = train_model(model, train_dataloader, val_dataloader)

                val_acc = trainer.callback_metrics["val_acc"].item()
                val_accs.append(val_acc)

            avg_val_acc = np.mean(val_accs)
            if avg_val_acc > best_logreg_val_acc:
                best_logreg_val_acc = avg_val_acc
                best_logreg_model = model
                best_logreg_param = {"penalty": penalty, "optimizer_fn": optimizer_fn.__name__, "learning_rate": learning_rate, "weight_decay": weight_decay}

        # 10. Predizione e valutazione per la regressione logistica
        y_pred_logreg = best_logreg_model(X_test).detach().cpu().numpy().round()
        accuracy_logreg = accuracy_score(y_test.cpu().numpy(), y_pred_logreg)
        f1_logreg = f1_score(y_test.cpu().numpy(), y_pred_logreg)
        auc_logreg = roc_auc_score(y_test.cpu().numpy(), y_pred_logreg)
        results.append({
            "Model": "Logistic Regression",
            "Dataset": i + 1,
            "Best_Params": best_logreg_param,
            "Test_Accuracy": accuracy_logreg,
            "Test_F1": f1_logreg,
            "Test_AUC": auc_logreg,
            "Val_Accuracy": best_logreg_val_acc
        })

    print(tabulate(results, headers="keys", tablefmt="pretty"))


if __name__ == "__main__":
    exhaustive_search()
