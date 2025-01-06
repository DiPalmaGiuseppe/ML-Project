import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from scipy.stats import uniform, loguniform
import matplotlib.pyplot as plt
from utils import scorer, read_tr, read_ts, euclidean_distance_score, scale_data, save_figure, write_blind_results

X_train, X_test, y_train, y_test = read_tr(split = 0.2)
X_blind = read_ts()

X_train, X_test, X_blind, y_train, feature_scaler, target_scaler = scale_data(
    X_train, X_test, X_blind, y_train
)

svr = SVR()
multi_output_svr = MultiOutputRegressor(svr)

param_dist = {
    'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'estimator__C': loguniform(1e-3, 1e3),
    'estimator__gamma': ['scale', 'auto'],
    'estimator__degree': [2, 3, 4, 5],
    'estimator__epsilon': uniform(0.01, 0.1)
}


kf = KFold(n_splits=10, shuffle=True, random_state=42)
randomized_search = RandomizedSearchCV(multi_output_svr, param_distributions=param_dist, n_iter=500, scoring=scorer, cv=kf, verbose=1, n_jobs=-1)
randomized_search.fit(X_train, y_train)

# Migliori parametri
print("Best parameters:", randomized_search.best_params_)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
best_model = randomized_search.best_estimator_
best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
train_loss = euclidean_distance_score(y_train, y_train_pred)
print("Train Loss:", train_loss)

y_val_pred = best_model.predict(X_val)
val_loss = euclidean_distance_score(y_val, y_val_pred)
print("Val Loss:", val_loss)


y_pred = best_model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred)

test_loss = euclidean_distance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("---- Final Results (original scale data) ----")
print("Test Loss:", test_loss)
print("Test R²:", r2)

y_blind_pred = best_model.predict(X_blind)
y_blind_pred = target_scaler.inverse_transform(y_blind_pred)
write_blind_results("SklearnSVM", y_blind_pred)