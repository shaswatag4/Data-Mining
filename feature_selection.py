import pandas as pd
import itertools

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_rows = 999
pd.set_option('display.max_colwidth', -1)


def svc_param_selection(X, y, nfolds):
    from sklearn import svm

    GridSearchCV = sklearn.model_selection.GridSearchCV
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


data_path = r'./data/game_data/'

# Evaluation set
# Training set
years = ['2010',
         '2011',
         '2012',
         '2013',
         '2014',
         '2015',
         '2016',
         '2017'
         ]

base_features = [
    'week',
    'home_wpct',
    'home_h_wpct',
    'home_prev_wpct',
    'away_wpct',
    'away_a_wpct',
    'away_prev_wpct',
    'matchup_weight',
]

additional_features = [
    'home_prev_h_wpct',
    'away_prev_a_wpct',
    'div_flag',
    'home_season_pt_dif',
    'home_3game_pt_dif',
    'home_5game_pt_dif',
    'home_prev_season_pt_dif',
    'away_season_pt_dif',
    'away_3game_pt_dif',
    'away_5game_pt_dif',
    'away_prev_season_pt_dif'
                 ]

all_files = [data_path + str(year) + '_database.csv' for year in years]
data_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

feature_sub_list = []
for r in range(6, len(additional_features)+1):
    [feature_sub_list.append(l) for l in list(itertools.combinations(additional_features, r))]

results = []
counter = 1
for f in feature_sub_list:

    additional_features = list(f)
    feature_list = base_features + additional_features

    if counter % 10 == 0:
        print '{}/{}'.format(counter, len(feature_sub_list)+1)
    counter += 1

    X = data_df[feature_list]
    Y = data_df['result']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.125, shuffle=False)

    svm = SVC()
    svm.fit(X_train, Y_train)
    train_accuracy = svm.score(X_train, Y_train)
    test_accuracy = svm.score(X_test, Y_test)

    results.append([train_accuracy, test_accuracy, additional_features])

df = pd.DataFrame(results, columns=['train_accuracy', 'test_accuracy', 'features'])

sorted_df = df.sort_values(by='test_accuracy', ascending=False)

# print sorted_df
print sorted_df.head(20).to_string()

# row = df.loc[df['test_accuracy'].idxmax()]
# print row['train_accuracy'], row['test_accuracy'],  row['features']
