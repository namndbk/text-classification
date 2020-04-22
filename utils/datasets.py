from ftfy import fix_text
from random import shuffle
import numpy as np
import pickle
import os


from preprocessing import normalize


def load_dataset(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    X = []
    y = []
    for item in data:
        X.append(normalize(item["text"]))
        y.append(item["label"])
    del(data)
    return X, y


def load_data(folder):
    data = []
    label = folder.lower().replace(" ", "_")
    files = os.listdir(folder)
    for file_name in files:
        file_name = folder + "/" + file_name
        with open(file_name, "rb") as f:
            line = f.read()
            line = line.decode("utf-16")
        data.append({"label": label, "text": fix_text(line)})
    return data


def dump_data(file_name, data):
    corpus = []
    tags = list(set(element["label"] for element in data))
    labels = {t: i for i, t in enumerate(tags)}
    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    for e in data:
        corpus.append({"label": labels[e["label"]], "text": e["text"]})
    del(data)
    shuffle(corpus)
    with open(file_name, "wb") as f:
        pickle.dump(corpus, f)


# if __name__ == "__main__":
#     current_path = os.getcwd()
#     train_path = os.path.join(current_path, "data", "train_full")
#     test_path = os.path.join(current_path, "data", "test_full")
#     data_train = []
#     data_test = []
#     for folder in os.listdir(train_path):
#         folder = os.path.join(train_path, folder)
#         data = load_data(folder)
#         for x in data:
#             data_train.append(x)
#     dump_data("train", data_train)
#     del(data_train)
#     for folder in os.listdir(test_path):
#         folder = os.path.join(test_path, folder)
#         data = load_data(folder)
#         for x in data:
#             data_test.append(x)
#     dump_data("test", data_test)
#     del(data_test)
