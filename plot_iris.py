import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pickle
import seaborn as sns; sns.set(font_scale=1.2)
import pandas as pd
from joblib import dump, load
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
            for i in range(1975, 2101):
                filename = 'mal_7/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'benign_7':
            for i in range(11, 1310):
                filename = 'benign_7/ldscan' + str(i) + '.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

        if name == 'current':
            if os.path.getsize('../Botvac-Control/ldscan_current.pkl') > 0:
                filename = '../Botvac-Control/ldscan_current.pkl'
                scan = read_file(filename)
                self._scans.append(scan)

# def make_meshgrid(x, y, h=.02):
#     """Create a mesh of points to plot in
#
#     Parameters
#     ----------
#     x: data to base x-axis meshgrid on
#     y: data to base y-axis meshgrid on
#     h: stepsize for meshgrid, optional
#
#     Returns
#     -------
#     xx, yy : ndarray
#     """
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy
#
#
# def plot_contours(ax, clf, xx, yy, **params):
#     """Plot the decision boundaries for a classifier.
#
#     Parameters
#     ----------
#     ax: matplotlib axes object
#     clf: a classifier
#     xx: meshgrid ndarray
#     yy: meshgrid ndarray
#     params: dictionary of params to pass to contourf, optional
#     """
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out
#
#
# # import some data to play with
# iris = datasets.load_iris()
# # Take the first two features. We could avoid this by using a two-dim dataset
# X = iris.data[:, :2]
# y = iris.target


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
# models = (svm.SVC(kernel='linear', C=C),
#           svm.LinearSVC(C=C),
#           svm.SVC(kernel='rbf', gamma=0.7, C=C),
#           svm.SVC(kernel='poly', degree=3, C=C))
# models = svm.SVC(kernel='linear', C=C)
# models = models.fit(X, y)

# title for the plots
# title = 'SVC with linear kernel'
#
# # Set-up 2x2 grid for plotting.
# fig, sub = plt.subplots(2, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

data = Data()
data.read_data('benign_7')
train_ = []
for i in data._scans:
    train_angle = []
    train_angle.append(1)
    for j in range(0, 360):
        if int(i.errors[j]) > 0:
            train_angle.append(1)
        else:
            train_angle.append(0)
    train_.append(train_angle)

data1 = Data()
data1.read_data('mal_7')
for i in data1._scans:
    train_angle = []
    train_angle.append(0)
    for j in range(0, 360):
        if int(i.errors[j]) > 0:
            train_angle.append(1)
        else:
            train_angle.append(0)
    train_.append(train_angle)

#create lable
labels = []
labels.append('Label')
for i in range(0,360):
    labels.append('Angle'+str(i))
dataset = pd.DataFrame.from_records(train_,columns=labels)

X = dataset[labels[1:]].as_matrix()
y = np.where(dataset['Label']==1, 0, 1)

dataset.to_csv('dataset.csv', sep='\t')

# Feature names
features = dataset.columns.values[1:].tolist()

clf = svm.SVC(kernel='linear')
clf.fit(X, y)
dump(clf, 'train.joblib')


# Get the separating hyperplane
w = clf.coef_[0]
print(w)
a = -w[5] / w[1]
xx = np.linspace(30, 60)

yy = a * xx - clf.intercept_[0] / w[5]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[300] - a * b[299])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[300] - a * b[299])

# sns.lmplot('Angle299', 'Angle300', data=dataset, hue='Label', palette='Set1', fit_reg=False, scatter_kws={"s": 300})
# plt.plot(xx, yy, linewidth=2, color='black')

# # X = [[1,2,3,4,3,4],[4,5,4,4,5,6]]
# # y = [1,0]
# # y = np.array(y)
# # X = np.array(X)
# print(len(X))
# X = X[:,0]
# xx, yy = make_meshgrid(X, y)
#
# ax = sub.flatten()[0]
# plot_contours(ax, models, xx, yy,
#               cmap=plt.cm.coolwarm, alpha=0.8)
# if len(X)==len(y):
#     print('ok')
# ax.scatter(X, y, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)


# Look at the margins and support vectors
sns.lmplot('Label', 'Angle5', data=dataset, hue='Label', palette='Set1', fit_reg=False, scatter_kws={"s": 1})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none');
plt.show()
