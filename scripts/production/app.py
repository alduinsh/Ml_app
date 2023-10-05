
from flask import Flask, jsonify, abort
import pickle
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return "Прогнозируем уровень заработной платы вместе!"

@app.route('/predict/<int:experience_level>/<int:job_title>')
def predict(Feature1,job_title):
    try:
        Feature1 = int(Feature1)
      job_title = int(job_title)
    except ValueError:
        abort(400, 'Данные не верны')

    try:
        with open('../models/model.pkl', 'rb') as fd:
            clf = pickle.load(fd)
        prediction = int(clf.predict([[Feature1,job_title]])[0])
        return jsonify({'Target': prediction}), 200, {'Content-Type': 'application/json'}
    except FileNotFoundError:
        abort(404, "Модель не найдена")
    except Exception:
        abort(500, "Внутренняя ошибка сервера")

if __name__ == '__main__':
    app.run()
