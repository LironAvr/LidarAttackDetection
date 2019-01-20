# -*- coding: utf-8 -*-
from flask import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import time, os
from joblib import dump, load

IGNORE_ANGLES = [25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                 85, 86, 87, 88, 89, 90, 91, 97, 98, 122, 126, 127, 128, 129, 130, 131, 132, 133, 136, 137, 139, 140,
                 141, 142, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
                 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183,
                 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
                 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
                 246, 247, 248, 264, 279, 280, 281, 296, 297, 298, 299, 306, 307, 308, 309, 310, 311, 312, 315, 325,
                 326, 327, 328, 329, 330, 331, 338, 339, 340, 341, 55, 66, 135, 138, 143, 146, 178, 227, 235, 278, 295,
                 249, 134, 253, 42, 37, 93, 305, 290, 294, 266, 32, 300, 332, 314, 283, 68, 43, 343, 0, 342, 251, 44,
                 49, 2, 34, 282, 333, 268, 301, 67, 33, 250, 92, 302, 74, 334, 100, 313, 36, 45, 293, 324, 3, 95, 317,
                 4, 319, 99, 124, 35, 303, 316, 336, 96, 101, 255, 286, 304]

IGNORE_ANGLES1 = []

IGNORE_ANGLES2 = []

IGNORE_ANGLES = IGNORE_ANGLES + IGNORE_ANGLES1 + IGNORE_ANGLES2


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

        if name == 'mal_6':
            for i in range(540, 761):
                filename = 'mal_6/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_7':
            for i in range(1975, int(1975 + (2101 - 1975) / 2)):
                filename = 'mal_7/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'mal_7_test':
            for i in range(int(1975 + (2101 - 1975) / 2), 2101):
                filename = 'mal_7/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_7':
            for i in range(11, int(11 + (1310 - 11) / 2)):
                filename = 'benign_7/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_7_test':
            for i in range(int(11 + (1310 - 11) / 2), 1310):
                filename = 'benign_7/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'current':
            if os.path.getsize('../Botvac-Control/ldscan_current.pkl') > 0:
                filename = '../Botvac-Control/ldscan_current.pkl'
                scan = read_file(filename)
                self._scans.append(scan)


def train():
    data = Data()
    data.read_data('benign_7')
    train_x = []
    train_y = []
    for i in data._scans:
        train_angle = []
        for j in range(0, 360):
            if int(i.errors[j]) > 0:
                train_angle.append(1)
            else:
                train_angle.append(0)
        train_x.append(train_angle)
        train_y.append(0)
    data1 = Data()
    data1.read_data('mal_7')
    for i in data1._scans:
        train_angle = []
        for j in range(0, 360):
            if int(i.errors[j]) > 0:
                train_angle.append(1)
            else:
                train_angle.append(0)
        train_x.append(train_angle)
        train_y.append(1)
    X = np.array(train_x)
    y = np.array(train_y)
    # clf = RandomForestClassifier(max_depth=100, random_state=0)  # max_depth is set max
    clf = svm.SVC()
    clf.fit(X, y)
    dump(clf, 'train.joblib')
    # def plot_decision_function(classifier, sample_weight, axis, title):
    #     # plot the decision function
    #     xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    #     print(xx)
    #     Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #
    #     # plot the line, the points, and the nearest vectors to the plane
    #     axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    #     axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
    #                  cmap=plt.cm.bone, edgecolors='black')
    #
    #     axis.axis('off')
    #     axis.set_title(title)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # sample_weight_constant = np.ones(len(X))
    # plot_decision_function(clf,sample_weight_constant, axes[0],"Constant weights")
    # plt.show()
    print("Accuracy when training: " + str(clf.score(X, y)))
    print("clf" + str(clf))

    # #draw tree
    # import IPython.display, graphviz, re, io
    # from matplotlib.pyplot import imread
    # from sklearn.tree import DecisionTreeClassifier, export_graphviz
    # import pydotplus
    #
    # def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    #     s = export_graphviz(t, out_file=None, feature_names='o', filled=True,
    #                         special_characters=True, rotate=True, precision=precision)
    #     IPython.display.display(graphviz.Source(re.sub('Tree {',
    #                                                    f'Tree {{ size={size}; ratio={ratio}', s)))
    #
    # def show_tree(tree, features, path):
    #     f = io.StringIO()
    #     export_graphviz(tree, out_file=f, feature_names=['percentage'])
    #     pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    #     img = imread(path)
    #     plt.rcParams['figure.figsize'] = [20, 20]
    #     plt.imshow(img)
    #
    # # draw_tree(clf.estimators_[0], X, precision=3)
    # show_tree(clf.estimators_[4], '', 'test.jpg')

    return clf


def classify(scan, train_result):
    data_x = []
    clf = train_result
    train_angle = []
    for i in range(0, 360):
        if int(scan.errors[i]) > 0:
            train_angle.append(1)
        else:
            train_angle.append(0)
    data_x.append(train_angle)
    data_x = np.array(data_x)
    result = clf.predict(data_x)  # Get predict results for each angle.
    # If there have more than 50% benign signature for each angle, this scan will be benign.
    # Malicious scans are the same
    if result[0] == 0:
        return False
    elif result[0] == 1:
        return True


train_result = train()
test_x = []
test_y = []
total = 0
correct = 0

# For test
data1 = Data()
data1.read_data('benign_7_test')

for i in data1._scans:
    total += 1
    result = classify(i, train_result)
    if result == False:
        correct += 1

data2 = Data()
data2.read_data('mal_7_test')
for i in data2._scans:
    total += 1
    result = classify(i, train_result)
    if result == True:
        correct += 1

print("Accuracy for classification: " + str(correct * 100 / total) + "%")

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