from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Caricamento del dataset
monk_s_problems = fetch_ucirepo(id=70)

X = monk_s_problems.data.features 
y = monk_s_problems.data.targets 

# Divisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definizione del modello
svm = SVC()

# Definizione dei possibibili iper-parametri
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search degli iper-parametri con validazione incrociata k-fold (ad esempio, k=5)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Training del mmodello
y_train = y_train.values.ravel()
grid_search.fit(X_train, y_train)

print("Migliori parametri:", grid_search.best_params_)

# Selezione del modello migliore
best_model = grid_search.best_estimator_

# Esegui la previsione sul dataset di test
y_pred = best_model.predict(X_test)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print("Accuratezza sul test dataset:", accuracy)
