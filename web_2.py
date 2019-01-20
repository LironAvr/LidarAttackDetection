# -*- coding: utf-8 -*-
from flask import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import time,os


class LidarScan:

    def __init__(self, ldscan):

        self.distances = []
        self.intensities = []
        self.errors = []

        for i in range(0, len(ldscan)):
            response = ldscan[i].strip()
            if len(response) < 1:
                break
            lst = response.split(',')
            if len(lst) < 4:
                continue
            if lst[0].lower() == 'AngleInDegrees'.lower():
                continue
            if lst[0] == 'ROTATION_SPEED':
                self.rotation = float(lst[1])
                continue
            angle = int(lst[0])
            if -1 < angle < 360 and len(lst) > 2:
                self.distances.append(int(lst[1]))
                self.intensities.append(int(lst[2]))
                self.errors.append(int(lst[3]))
                continue
            print(self)


def read_file(filename):
    output = open(filename, 'rb')
    a = pickle.load(output)
    output.close()
    return a

class Data:
    def __init__(self):
        self.benign_scans = []
        self.malicious_scans = []
        self._scans = []

    def read_data(self, name):


        if name == 'reflective':
            for i in range(379, 1152):
                filename = 'reflective/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_s':
            for i in range(0, 303):
                filename = 'benign_s/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_s':
            for i in range(305, 550):
                filename = 'mal_s/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_2':
            for i in range(0, 108):
                filename = 'benign_2/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_2':
            for i in range(109, 226):
                filename = 'mal_2/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_1':
            for i in range(0, 113):
                filename = 'mal_1/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_1':
            for i in range(114, 223):
                filename = 'benign_1/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_3':
            for i in range(13, 124):
                filename = 'mal_3/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_4':
            for i in range(719, 943):
                filename = 'benign_4/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_4':
            for i in range(944, 1168):
                filename = 'mal_4/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'current':
            if os.path.getsize('../Botvac-Control/ldscan_current.pkl') > 0:
                filename = '../Botvac-Control/ldscan_current.pkl'
                scan = read_file(filename)
                self._scans.append(scan)


def train():
    data = Data()
    data.read_data('benign_4')
    train_x = []
    train_y = []
    for i in data._scans:
        for j in range(0, 359):
            train_x.append([i.distances[j], i.errors[j], i.intensities[j]])
            train_y.append(1)
    data1 = Data()
    data1.read_data('mal_4')
    for i in data1._scans:
        for j in range(0, 359):
            train_x.append([i.distances[j], i.errors[j], i.intensities[j]])
            train_y.append(0)
    X = np.array(train_x)
    y = np.array(train_y)
    clf = RandomForestClassifier(max_depth=100, random_state=0)  # max_depth is set max
    clf.fit(X, y)
    print("Accuracy when training: " + str(clf.score(X, y)))
    print("clf" + str(clf))
    return clf

def classify(scan, train_result):
    data_x = []
    clf = train_result

    for i in range(0, 360):
        data_x.append([scan.distances[i], scan.errors[i], scan.intensities[i]])
    data_x = np.array(data_x)
    result = clf.predict(data_x)  # Get predict results for each angle.
    # If there have more than 50% benign signature for each angle, this scan will be benign.
    # Malicious scans are the same
    benign = 0
    mal = 0
    for i in result:
        if i == 1:
            benign += 1
        elif i == 0:
            mal += 1
    print(benign)
    # print(mal)
    if benign > 10:
        return True
    else:
        return False
    # elif mal >= benign:
    #     return False


train_result =  train()
test_x = []
test_y = []
total = 0
correct = 0



    # data = Data()
    # data.read_data('benign_1')
    # data.read_data('benign_2')
    # for i in data._scans:
    #     total += 1
    #     result = classify(i, train_result)
    #     if result == True:
    #         correct += 1
    #
    # data1 = Data()
    # data1.read_data('mal_3')
    # for i in data1._scans:
    #     total += 1
    #     result = classify(i, train_result)
    #     if result == False:
    #         correct += 1
    #
    # print("Accuracy for classification: " + str(correct * 100 / total) + "%")
# Main program
# Define web
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "fasdfsdfsdfasdfs"
# app update
jinja_options = app.jinja_options.copy()
jinja_options.update(dict(
    variable_start_string='[[',
    variable_end_string=']]'
))
app.jinja_options = jinja_options

# Routing index
@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


# API Lấy thông tin lidar
@app.route('/get_data_api', methods=['POST', 'GET'])
def get_data_api():
    if request.method == 'POST':
        time.sleep(0.5)
        data = Data()
        data.read_data('current')
        for i in data._scans:
            result = classify(i, train_result)
            if result == True:
                print('Benign!')
            elif result == False:
                print('Mal!')
        return json.dumps(result)
    return 'nothing'


# App run
app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
SESSION_TYPE = 'redis'
app.config.from_object(__name__)
Session(app)