import pandas as pd
import numpy as np

# Define parameters for the mock dataset
trials_per_model = 20
models = ["Logistic Regression", "Naive Bayes", "Decision Tree", "SVM"]
num_records = trials_per_model * len(models)

# Generate mock data
data = {
    "Trial": np.tile(np.arange(1, trials_per_model + 1), len(models)),
    "Model": np.repeat(models, trials_per_model),
    "Training_Time_Seconds": np.random.uniform(50, 500, num_records),  # Random values between 5 and 50 seconds
    "Training_Energy_Joules": np.random.uniform(100, 500, num_records),  # Random values between 100 and 500 joules
    "Inference_Time_Seconds": np.random.uniform(0.01, 0.1, num_records),  # Random values between 0.01 and 0.1 seconds
    "Inference_Energy_Joules": np.random.uniform(1, 50, num_records),  # Random values between 1 and 10 joules
    "F1_Score": np.random.uniform(0.5, 1, num_records)  # Random values between 0.5 and 1 for F1 Score
}

# Create DataFrame
df_mock = pd.DataFrame(data)

# Save as CSV
df_mock.to_csv("mock_experiment_data.csv", index=False)
