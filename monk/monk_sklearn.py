from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from monk_dataset import monk_loader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore', message=".*total space of parameters.*")

def test_grid_search(model, param_grid):
    for i in range(3):
        X_train, y_train, X_test, y_test = monk_loader(i + 1)

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=19)

        # GridSearchCV setup
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=stratified_kfold,
            scoring='accuracy',
            n_jobs=-1
        )

        # Fit the GridSearchCV
        grid_result = grid_search.fit(X_train, y_train)

        # Print results
        print(f"Dataset Monk-{i+1}")
        print("Best Parameters:", grid_search.best_params_)

        # Best model
        best_model = grid_search.best_estimator_

        # Final prediction on test data
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Test set accuracy:", accuracy)

# MLP Classifier
print("---- MLP Classifier ----")
mlp = MLPClassifier(
    max_iter=500, 
    early_stopping=True,
    n_iter_no_change=10,
    tol=1e-4,
    random_state=19
)
mlp_param_grid = {
    'hidden_layer_sizes': [(2,), (3,), (4,)],
    'momentum': [0.7, 0.75, 0.8, 0.85, 0.9],
    'learning_rate_init': [0.01, 0.1, 0.2]
}

test_grid_search(mlp, mlp_param_grid)