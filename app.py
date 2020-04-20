from flask import Flask, request, render_template
from flask import json, jsonify
from models import Classifier
from utils.preprocessing import normalize

from sklearn.svm import LinearSVC

import pickle


app = Flask(__name__)


models = Classifier()
models.load_model("model/colabs/model_v1")
with open("model/colabs/transform", "rb") as f:
    transform = pickle.load(f)


def predict(text):
    text = normalize(text)
    text = transform.transform([text])
    label = models.estimator.predict(text)
    out = label[-1].replace("_", " ").title()
    return out


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET","POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        doc = request.form["document"]
        out = predict(doc)
        return render_template("index.html", document=doc, message=out)
        


@app.route("/api", methods=["GET", "POST"])
def api():
    if request.method == "GET":
        message = {
            "message": "hello",
            "text": ""
        }
    else:
        doc = request.get_json()["text"]
        out = predict(doc)
        message = {
            "message": out,
            "text": doc
        }
    return jsonify(message)


if __name__ == "__main__":
    app.run()