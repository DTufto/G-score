import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def preprocess_image(image):
    image = image.convert("RGB")  # Convert the image to RGB color space
    image = image.resize((64, 64))  # Resize to a smaller size for faster processing
    return np.array(image).flatten()  # Flatten the image to a 1D array


def get_train_test_split(test_size=0.2, random_state=42):
    dataset = load_dataset("microsoft/cats_vs_dogs")

    X = np.array([preprocess_image(item['image']) for item in dataset['train']])
    y = np.array([item['labels'] for item in dataset['train']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
