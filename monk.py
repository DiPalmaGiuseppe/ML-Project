from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from monk_dataset import monk_loader
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
from sklearn.model_selection import StratifiedKFold
import warnings
import random
warnings.filterwarnings('ignore', message=".*total space of parameters.*")

def test_random_search(model, param_dist):
    for i in range(3):
        X_train, y_train, X_test, y_test = monk_loader(i+1)

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        penalty = ['l2', 'l1']

        if isinstance(model, LogisticRegression):
            if random.choice(penalty) == 'l2':
                param_dist["penalty"] = ['l2']
                param_dist["solver"] = ['liblinear', 'saga', 'newton-cg', 'lbfgs'] 
            else:
                param_dist["penalty"] = penalty
                param_dist["solver"] = ['liblinear', 'saga']

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                           n_iter=500, cv=stratified_kfold, scoring='accuracy', 
                                           random_state=42, n_jobs=-1)
        
        random_search.fit(X_train, y_train)

        print("Best Parameters:", random_search.best_params_)
        
        best_model = random_search.best_estimator_

        # Final prediction on test data
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Test set accuracy:", accuracy)

# SVM Classifier
print("---- SVM ----")
svm = SVC(random_state=42)
svm_param_dist = {
    'C': uniform(0.1, 1000),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': uniform(0.001, 10000)
}
test_random_search(svm, svm_param_dist)

# MLP Classifier
print("---- MLP Classifier ----")
mlp = MLPClassifier(random_state=42, max_iter=10000, early_stopping=True)
mlp_param_dist = {
    'hidden_layer_sizes': [(1,), (2,), (5,), (10,),(2,2), (5,5), (10,10)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': uniform(0.0001, 0.1),
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}
test_random_search(mlp, mlp_param_dist)

# Logistic Regression
print("---- Logistic Regression ----")
logreg = LogisticRegression(random_state=42, max_iter=10000)
logreg_param_dist = {
    'C': uniform(0.01, 1000),
    'tol': uniform(1e-4, 1e-2),
    'class_weight': [None, 'balanced'],
    'warm_start': [True, False],
    'intercept_scaling': uniform(1,1000)
}
test_random_search(logreg, logreg_param_dist)

# Decision Tree Classifier
print("---- Decision Tree ----")
dt = DecisionTreeClassifier(random_state=42)
dt_param_dist = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
}

test_random_search(dt, dt_param_dist)