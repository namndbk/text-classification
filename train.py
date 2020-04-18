import argparse
import os
import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


from utils.datasets import load_dataset
from utils import *


from models import Classifier


X_train, y_train = load_dataset("train")
X_test, y_test = load_dataset("test")

transformer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5)
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

est = LinearSVC()

models = Classifier(estimator=est)
models.fit(X_train, y_train)

y_pred = models.predict(X_test)

print("Acc: %.2f %" % (accuracy_score(y_test, y_pred)*100))
print("F1 score: %.2f %" % (f1_score(y_test, y_pred)))


with open("./model/transformer.pkl", "wb") as f:
    pickle.dump(transformer, f)

models.save_model("./model/model_ver1.pkl")
