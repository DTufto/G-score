import time
import csv
import pyRAPL
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from data_processing import get_train_test_split
from datetime import datetime
import os
import numpy as np


class EnergyMeasurement:
    def __init__(self):
        # Initialize PyRAPL
        try:
            pyRAPL.setup()
            self.measurement = pyRAPL.Measurement('ML Models')
            self.measurement_available = True
        except Exception as e:
            print(f"Warning: PyRAPL initialization failed: {e}")
            print("Energy measurements will not be available.")
            self.measurement_available = False

    def measure_energy(self, func, *args, **kwargs):
        """
        Measure both time and energy consumption of a function
        Returns: (execution_time, energy_consumption)
        """
        start_time = time.time()

        if self.measurement_available:
            try:
                self.measurement.begin()
                result = func(*args, **kwargs)
                self.measurement.end()
                energy = round(sum(self.measurement.result.pkg) / 1e6, 2)
            except Exception as e:
                print(f"Warning: Energy measurement failed: {e}")
                energy = None
                result = func(*args, **kwargs)
        else:
            energy = None
            result = func(*args, **kwargs)

        execution_time = round(time.time() - start_time, 2)
        return result, execution_time, energy


def write_result_to_csv(result, csv_filename, is_first_write=False):
    """
    Write a single result to the CSV file
    """
    fieldnames = ['Trial', 'Model', 'Training_Time_Seconds', 'Training_Energy_Joules',
                  'Inference_Time_Seconds', 'Inference_Energy_Joules', 'F1_Score']

    file_exists = os.path.isfile(csv_filename)
    mode = 'a' if file_exists else 'w'

    with open(csv_filename, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def get_svm_configuration(X_train):
    """Get SVM-specific configuration parameters"""
    n_samples, n_features = X_train.shape
    use_dual = n_samples < n_features
    sqrt_features = int(np.sqrt(n_features))
    max_features_ratio = sqrt_features / n_features
    return use_dual, max_features_ratio


def prepare_data_for_model(X_train, X_test, model_name):
    """Prepare data based on model requirements"""
    if model_name == "Support Vector Machine" or model_name == "Logistic Regression":
        # Apply scaling only for SVM
        scaler = StandardScaler()
        X_train_prepared = scaler.fit_transform(X_train)
        X_test_prepared = scaler.transform(X_test)
    else:
        # Return original data for other models
        X_train_prepared = X_train
        X_test_prepared = X_test

    return X_train_prepared, X_test_prepared


def get_model_dict(X_train):
    """Create dictionary of models with appropriate configurations"""
    use_dual, max_features_ratio = get_svm_configuration(X_train)

    return {
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": CategoricalNB(),
        "Support Vector Machine": BaggingClassifier(
            LinearSVC(dual=use_dual),
            n_estimators=60,
            max_samples=0.05,
            max_features=max_features_ratio,
            bootstrap_features=True
        ),
    }


def run_experiments(num_trials):
    X_train, X_test, y_train, y_test = get_train_test_split()
    energy_measurer = EnergyMeasurement()
    models = get_model_dict(X_train)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'model_performance_results_{timestamp}.csv'

    for trial in range(num_trials):
        print(f"\nStarting trial {trial + 1}/{num_trials}")

        for model_name, clf in models.items():
            print(f"Processing {model_name}...")

            # Prepare data specifically for this model
            X_train_prepared, X_test_prepared = prepare_data_for_model(X_train, X_test, model_name)

            # Training phase
            _, training_time, training_energy = energy_measurer.measure_energy(
                clf.fit, X_train_prepared, y_train
            )

            # Inference phase
            y_pred, inference_time, inference_energy = energy_measurer.measure_energy(
                clf.predict, X_test_prepared
            )

            # Calculate F1 score
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Store and immediately write results
            result = {
                'Trial': trial + 1,
                'Model': model_name,
                'Training_Time_Seconds': training_time,
                'Training_Energy_Joules': training_energy if training_energy is not None else 'N/A',
                'Inference_Time_Seconds': inference_time,
                'Inference_Energy_Joules': inference_energy if inference_energy is not None else 'N/A',
                'F1_Score': round(f1, 4)
            }

            write_result_to_csv(result, csv_filename)

            print(f"Results for {model_name}:")
            print(f"  Training Time: {training_time}s")
            print(f"  Training Energy: {training_energy if training_energy is not None else 'N/A'} J")
            print(f"  Inference Time: {inference_time}s")
            print(f"  Inference Energy: {inference_energy if inference_energy is not None else 'N/A'} J")
            print(f"  F1 Score: {round(f1, 4)}")

    print(f"\nAll results have been saved to {csv_filename}")


if __name__ == "__main__":
    run_experiments(num_trials=20)