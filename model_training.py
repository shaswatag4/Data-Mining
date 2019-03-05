import utils
import nflgame
import sklearn
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import itertools

pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_rows = 999

feature_names = [
    'week',
    'home_wpct',
    'home_h_wpct',
    'home_prev_wpct',
    'home_prev_h_wpct',
    'away_wpct',
    'away_a_wpct',
    'away_prev_wpct',
    'away_prev_a_wpct',
    'div_flag',
    'matchup_weight',
    'home_season_pt_dif',
    'home_3game_pt_dif',
    'home_5game_pt_dif',
    'home_prev_season_pt_dif',
    'away_season_pt_dif',
    'away_3game_pt_dif',
    'away_5game_pt_dif',
    'away_prev_season_pt_dif'
                 ]

years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

data_path = r'./data/game_data/'
all_files = [data_path + str(year) + '_database.csv' for year in years]
data_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# Remove unknown results and ties from the data
data_df = utils.scrub_data(data_df)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

features_to_scale = ['home_season_pt_dif',
                     'home_3game_pt_dif',
                     'home_5game_pt_dif',
                     'home_prev_season_pt_dif',
                     'away_season_pt_dif',
                     'away_3game_pt_dif',
                     'away_5game_pt_dif',
                     'away_prev_season_pt_dif']

data_df[features_to_scale] = scaler.fit_transform(data_df[features_to_scale])

X = data_df[feature_names]
Y = data_df['result']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=True)


def svc_param_selection(X, y, nfolds):
    from sklearn import svm

    GridSearchCV = sklearn.model_selection.GridSearchCV
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


print '---------'
params = svc_param_selection(X_train, Y_train, 5)
svm = SVC(C=params['C'], gamma=params['gamma'])
trained_model = svm.fit(X_train, Y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
      .format(svm.score(X_train, Y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
      .format(svm.score(X_test, Y_test)))
y_pred = trained_model.predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

# class_names = trained_model.classes_
class_names = sorted(trained_model.classes_)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True, classes=class_names,
                      title='Normalized confusion matrix')

plt.show()
