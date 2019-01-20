# -*- coding: utf-8 -*-
from flask import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import time, os
import IPython.display, graphviz, re, io
from matplotlib.pyplot import imread
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

train_x = [[9, 9, 9, 9],
           [4, 4, 4, 4],
           [3, 3, 3, 3],
           [2, 2, 2, 2],
           [5, 5, 5, 5]]
train_y = [9,4,3,2,5]
X = np.array(train_x)
y = np.array(train_y)
clf = RandomForestClassifier(max_depth=100, random_state=2, n_estimators=10000)
clf.fit(X, y)
print("Accuracy when training: " + str(clf.score(X, y)))
print("clf" + str(clf))


def show_tree(tree, features, path):
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=['a', 'b', 'c','d'])
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imread(path)
    plt.rcParams['figure.figsize'] = [20, 20]
    plt.imshow(img)




result = clf.predict([[9,9,4,4]])
show_tree(clf.estimators_[1000], '', 'test1.jpg')
print(result)