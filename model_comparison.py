import utils
import nflgame
import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing


pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_rows = 999

data_path = r'./data/game_data/'

# Training set
years = ['2010', '2011', '2012', '2013', '2014', '2015']
all_files = [data_path + str(year) + '_database.csv' for year in years]
train_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
train_df = utils.scrub_data(train_df)

# Testing set
years = ['2016', '2017']
all_files = [data_path + str(year) + '_database.csv' for year in years]
test_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
test_df = utils.scrub_data(test_df)

# Feature scaling
features_to_scale = ['home_season_pt_dif',
         'home_3game_pt_dif',
         'home_5game_pt_dif',
         'home_prev_season_pt_dif',
         'away_season_pt_dif',
         'away_3game_pt_dif',
         'away_5game_pt_dif',
         'away_prev_season_pt_dif']

from sklearn.preprocessing import scale

train_df[features_to_scale] = scale(train_df[features_to_scale])
test_df[features_to_scale] = scale(test_df[features_to_scale])

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

X_train = train_df[feature_names]
Y_train = train_df['result']

X_test = test_df[feature_names]
Y_test = test_df['result']

logreg = LogisticRegression(C=2)
logreg.fit(X_train, Y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(logreg.score(X_train, Y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test, Y_test)))


def knn_param_selection(X, y, nfolds):
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    GridSearchCV = sklearn.model_selection.GridSearchCV
    param_grid = {'n_neighbors': [5, 6, 7, 8, 9, 10],
                  'leaf_size': [1, 2, 3, 5],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    grid_search = GridSearchCV(knn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


params = knn_param_selection(X_train, Y_train, 3)
print params
knn = KNeighborsClassifier(**params)
knn.fit(X_train, Y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(X_train, Y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(X_test, Y_test)))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
      .format(lda.score(X_train, Y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
      .format(lda.score(X_test, Y_test)))

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
      .format(gnb.score(X_train, Y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
      .format(gnb.score(X_test, Y_test)))


def svc_param_selection(X, y, nfolds):
    from sklearn import svm
    import numpy as np

    GridSearchCV = sklearn.model_selection.GridSearchCV
    Cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear', 'rbf']
    param_grid = {'C': Cs, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


# Learning
params = svc_param_selection(X_train, Y_train, 3)
print params
svm = SVC(**params)
# svm = SVC()
svm.fit(X_train, Y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
      .format(svm.score(X_train, Y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
      .format(svm.score(X_test, Y_test)))
