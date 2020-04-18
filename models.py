import pickle
from time import time


class Classifier(object):

    def __init__(self, estimator=None):
        self.estimator = estimator


    def train(self, X_train, y_train):
        print("Training model")
        time_begin = time()
        self.estimator.fit(X_train, y_train)
        train_time = time() - time_begin
        print("Done !")
        print("train time: %0.3fs" % train_time)
    


    def test(self, X_test):
        print("Testing model")
        time_begin = time()
        y_pred = self.estimator.predict(X_test)
        test_time = time() - time_begin
        print("Done !")
        print("test time: %0.3f s" % test_time)
        return y_pred


    def save_model(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.estimator, f)
    


    def load_model(self, file_name):
        with open(file_name, "rb") as f:
            self.estimator = pickle.load(f)