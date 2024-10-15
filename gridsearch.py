import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_processing import get_train_test_split

print("Loading data...")
X_train, X_test, y_train, y_test = get_train_test_split()

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": CategoricalNB(),
    "Support Vector Machine": SVC()
}

param_grids = {
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "Naive Bayes": {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False]
    },
    "Support Vector Machine": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}

results = []

for model_name, clf in models.items():
    print(f"Training {model_name}...")
    param_grid = param_grids.get(model_name, {})

    print(f"Grid search for {model_name}...")
    grid_search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=5, n_jobs=5)

    print(f"Fitting {model_name}...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    print(f"Predicting {model_name}...")
    start_time = time.time()
    y_pred = grid_search.predict(X_test)
    end_time = time.time()

    prediction_time = end_time - start_time

    print(f"Evaluating {model_name}...")
    accuracy = accuracy_score(y_test, y_pred)

    result = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Training time: {training_time:.2f} seconds\n"
        f"Prediction time: {prediction_time:.2f} seconds\n"
        f"Best parameters: {grid_search.best_params_}\n"
        "----------------------------------------\n"
    )
    print(result)
    results.append(result)

with open('gridsearch_results.txt', 'w') as file:
    file.writelines(results)
