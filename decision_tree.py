import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_processing import get_train_test_split


# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = get_train_test_split()

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
