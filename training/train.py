import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_processing import get_train_test_split

X_train, X_test, y_train, y_test = get_train_test_split()

models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=2),
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": CategoricalNB(),
    "Support Vector Machine": SVC()
}

for model_name, clf in models.items():
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    start_time = time.time()
    y_pred = clf.predict(X_test)
    end_time = time.time()

    prediction_time = end_time - start_time

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy of {model_name}: {accuracy:.2f}")
    print(f"Training time of {model_name}: {training_time:.2f} seconds")
    print(f"Prediction time of {model_name}: {prediction_time:.2f} seconds")
