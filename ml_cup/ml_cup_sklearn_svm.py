import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from utils import scorer, read_tr, read_ts, euclidean_distance_score, scale_data

X_train, X_test, y_train, y_test = read_tr(split = 0.2)
X_blind = read_ts()

X_train, X_test, X_blind, y_train, feature_scaler, target_scaler = scale_data(
    X_train, X_test, X_blind, y_train
)

svr = SVR()
multi_output_svr = MultiOutputRegressor(svr)

param_grid = {
    'estimator__kernel': ['linear', 'rbf', 'poly'],
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1, 'scale'],
    'estimator__degree': [2, 3]
}

# GridSearchCV con K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(multi_output_svr, param_grid, cv=kf, scoring=scorer, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Migliori parametri
print("Best parameters:", grid_search.best_params_)
best_params = grid_search.best_params_

best_model = MultiOutputRegressor(SVR(
    kernel=best_params['estimator__kernel'],
    C=best_params['estimator__C'],
    gamma=best_params['estimator__gamma'],
    degree=best_params['estimator__degree']
))

# Suddivisione del training set in training e validation (80% training, 20% validation)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Parametro per il numero di epoche
epochs = 200  # Un numero elevato di epoche per il training

# Allenamento con early stopping
train_errors, val_errors = [], []
best_val_error = float('inf')
patience = 20  # Numero di epoche senza miglioramenti prima di fermarsi
counter = 0

for epoch in range(1, epochs + 1):
    best_model.fit(X_train_split, y_train_split)
    
    # Predizione sul training e validation set
    y_train_pred = best_model.predict(X_train_split)
    y_val_pred = best_model.predict(X_val_split)
    
    # Calcolo dell'errore
    train_error = euclidean_distance_score(y_train_split, y_train_pred)
    val_error = euclidean_distance_score(y_val_split, y_val_pred)
    
    train_errors.append(train_error)
    val_errors.append(val_error)
    
    # Early stopping: fermati se l'errore di validazione non migliora
    if val_error < best_val_error:
        best_val_error = val_error
        counter = 0  # Reset del counter
    else:
        counter += 1
    
    # Se il miglioramento si ferma, interrompi l'allenamento
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Valutazione sui dati di test
y_pred = best_model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred)

test_loss = euclidean_distance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Mean Euclidean Error:", test_loss)
print("Test R²:", r2)

# # Predizione sui dati blind
# y_blind_pred = best_model.predict(X_blind)
# y_blind_pred_rescaled = target_scaler.inverse_transform(y_blind_pred.reshape(-1, 1))

# # Salvataggio dei risultati
# np.savetxt("blind_results.csv", y_blind_pred_rescaled, delimiter=",")

# # Visualizzazione delle perdite
# plt.scatter(range(len(y_test_rescaled)), y_test_rescaled, label="True")
# plt.scatter(range(len(y_pred_rescaled)), y_pred_rescaled, label="Predicted")
# plt.title("True vs Predicted Values")
# plt.legend()
# plt.savefig("true_vs_predicted.png")
# plt.show()
