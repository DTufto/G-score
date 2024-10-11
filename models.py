import time

def train_model(model, X_train, y_train):
    start_time = time.time() # TODO remove
    clf = model()
    clf.fit(X_train, y_train)
    end_time = time.time() # TODO remove
    print(end_time - start_time) # TODO remove
    return clf
