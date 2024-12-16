from sklearn.neural_network import MLPClassifier
from monk_dataset import monk_loader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def test(id):
    # Carica i dati con normalizzazione
    X_train, y_train, X_test, y_test = monk_loader(id, normalize=True)

    # Inizializza il classificatore MLP
    mlpc = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, n_iter_no_change=25)

    # Griglia di parametri
    param_grid = {
        'hidden_layer_sizes': [(5,), (10,), (20,), (50,), (100,), (200,)],
        'activation': ['relu', 'logistic', 'identity', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.00001 ,0.0001, 0.001, 0.01, 0.1]
    }

    # Ricerca a griglia
    grid_search = GridSearchCV(estimator=mlpc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Migliori parametri:", grid_search.best_params_)

    # Punteggi di validazione durante la ricerca a griglia
    print("Punteggi di validazione durante la ricerca a griglia:")
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        print(f"Punteggio: {mean_score:.4f}, Parametri: {params}")
    
    best_model = grid_search.best_estimator_

    # Predizione finale sui dati di test
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuratezza finale sul test dataset:", accuracy)
    print(f"Punteggio di training migliore: {grid_search.best_score_:.4f}")

for i in range(3):
    test(i + 1)