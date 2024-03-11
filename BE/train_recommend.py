from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from train import train, get_test_authors
from recommend import recommend
import os
from query import get_all_authors

app = Flask(__name__)
CORS(app)

basedir = os.path.dirname((os.path.dirname(__file__)))
results_path = os.path.join(basedir, 'D:\Dongtacgia\dongtacgia\dongtacgia\Results')
@app.route('/')
def _index():
    return render_template('train_recommend.html')
    
@app.route('/get_test_authors', methods=['POST'])
def _get_test_authors():
    data_name = request.get_json()["data_name"]
    test_percent = request.get_json()["test_percent"]
    db_name = request.get_json()['db_name']
    return get_test_authors(data_name, test_percent, db_name)

@app.route('/train', methods=['POST'])
def _train():
    data_name = request.get_json()["data_name"]
    test_percent = request.get_json()["test_percent"]
    return train(data_name, test_percent)

@app.route('/recommend', methods=['POST'])
def _recommend():
    topic = request.get_json()['topic']
    from_date = request.get_json()["from_date"]
    to_date = request.get_json()["to_date"]
    author_id = request.get_json()['author_id']
    model_name = request.get_json()['model_name']
    return recommend(topic, from_date, to_date, author_id, model_name)

@app.route('/get_all_authors', methods=['POST'])
def _get_all_authors():
    topic = request.get_json()['topic']
    from_date = request.get_json()["from_date"]
    to_date = request.get_json()["to_date"]
    return get_all_authors(topic, from_date, to_date)

app.run(debug=True, host='127.0.0.1', port=5001)