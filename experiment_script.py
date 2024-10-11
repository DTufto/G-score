from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from models import train_model
from data_processing import get_train_test_split
import argparse
from datasets import load_dataset


MODELS = {
    'decision-tree': DecisionTreeClassifier,
    'logistic-regression': LogisticRegression,
    'support-vector-machine': SVC,
    'naive-bayes': CategoricalNB
}

def train_all_models(xtrain, ytrain):
    for model in MODELS:
        train_model(model, xtrain, ytrain)


def main(model_name):
    dataset = load_dataset("microsoft/cats_vs_dogs")
    if model_name not in MODELS:
        print(f'Model not supported, supported keys: {MODELS.keys()}')
        return

    model_class = MODELS[model_name]
    for _ in range(20):
        X_train, X_test, y_train, y_test = get_train_test_split(dataset)
        train_model(model_class, X_train, y_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a machine learning model.')
    parser.add_argument('model', type=str, help='Name of the model to run')
    args = parser.parse_args()
    main(args.model)
