import time
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
ds = load_dataset("microsoft/cats_vs_dogs")


# Function to preprocess images
def preprocess_image(image):
    image = image.convert("RGB")  # Convert the image to RGB color space
    image = image.resize((64, 64))  # Resize to a smaller size for faster processing
    return np.array(image).flatten()  # Flatten the image to a 1D array


# Extract features and labels
X = np.array([preprocess_image(item['image']) for item in ds['train']])
y = np.array([item['labels'] for item in ds['train']])

# X_test = np.array([preprocess_image(item['image']) for item in ds['test']])
# y_test = np.array([item['labels'] for item in ds['test']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Measure the time it takes to train the model
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time

# Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Training time: {training_time:.2f} seconds")
