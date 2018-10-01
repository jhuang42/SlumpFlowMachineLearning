# Requires Python3, scipy, matplotlib
# Run this program by running python3 slump_flow.py in terminal or running from this python file
import CleanData as cd
import RegressionSuite as rs
import pandas
import numpy
import random
import copy
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


def as_dictionary(data):
    # create a dictionary to use as input for our dataframe
    as_dict = dict()
    as_dict.update({'cement': data[..., 1]})
    as_dict.update({'slag': data[..., 2]})
    as_dict.update({'fly_ash': data[..., 3]})
    as_dict.update({'water': data[..., 4]})
    as_dict.update({'sp': data[..., 5]})
    as_dict.update({'coarse': data[..., 6]})
    as_dict.update({'fine': data[..., 7]})
    as_dict.update({'flow': data[..., 8]})
    return as_dict


def slump_flow():
    clean_data = cd.CleanData()
    reg_suite = rs.RegressionSuite()
    # x variables: {number, cement, slag, fly ash, water, superplasticizer,
    # coarse aggregate, fine aggregate} training data
    # y variable: slump flow
    # with x variables from 0:7 and slump flow in index 7
    data_as_list = clean_data.get_data()

    y_dict = dict()
    x_dict = dict()

    # Create 10 iterations of randomly selecting 85 observations for 5
    # fold cross validation (5 x 17; 4 x 17 training, 17 validate set)
    # Have the last 18 of the 103 be the observation test sets

    for i in range(10):
        # Randomly shuffle our data for random selection, ten times for 5-fold CV later on
        random_sample = copy.copy(data_as_list)
        random.shuffle(random_sample)
        dataframe = pandas.DataFrame(as_dictionary(numpy.array(random_sample)))
        df_1 = dataframe.iloc[:, 0:3]
        df_2 = dataframe.iloc[:, 4:]
        # separate the dataframe into one of the x variables and one of the slump flow
        x_dataframe = pandas.concat([df_1.reset_index(drop=True), df_2], axis=1)
        y_dataframe = dataframe.iloc[:, 3:4]
        # store the random data in a dict of each iteration with the key as the iteration number and value as the data
        x_dict.update({i: x_dataframe})
        y_dict.update({i: y_dataframe})

    # max score of cross validation in the coming loop, store in a tuple with first index as the score
    # and the second index as the key for the dicts
    max_cv_score = (0, 0, 0)
    max_cv_score_ridge = (0, 0)
    max_cv_score_lasso = (0, 0)

    best_x_test = None
    best_y_test = None
    # Task 0
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split\
            (x_dict.get(i), y_dict.get(i), test_size=0.168, random_state=0)
        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        fold_num = 1
        # Do the 5-fold CV
        for score in scores:
            if score > max_cv_score[0]:
                max_cv_score = (score, i, fold_num)
                best_x_test = X_test
                best_y_test = y_test
            fold_num += 1
        # Create separate dataframes for the validation and training sets
        # (5 x 17; 4 x 17 training, 17 validate set)
        X_valid = X_train[68:]
        y_valid = y_train[68:]
        X_train = X_train[:68]
        y_train = y_train[:68]
        ridge_score = reg_suite.get_ridge_score(X_train, y_train, X_valid, y_valid)
        if ridge_score > max_cv_score_ridge[0]:
            max_cv_score_ridge = (ridge_score, i)
        lasso_score = reg_suite.get_lasso_score(X_train, y_train, X_valid, y_valid)
        if lasso_score > max_cv_score_lasso[0]:
            max_cv_score_lasso = (lasso_score, i)

    # Best fit model training data for each of the three regressions
    max_score_x_data = x_dict.get(max_cv_score[1])
    max_score_y_data = y_dict.get(max_cv_score[1])

    max_score_x_data_ridge = x_dict.get(max_cv_score_ridge[1])
    max_score_y_data_ridge = y_dict.get(max_cv_score_ridge[1])

    max_score_x_data_lasso = x_dict.get(max_cv_score_lasso[1])
    max_score_y_data_lasso = y_dict.get(max_cv_score_lasso[1])

    # Best fit model test data for each of the regressions, OLS was test data was obtained above in the 5-fold CV
    best_x_test_ridge = x_dict.get(max_cv_score_ridge[1])
    best_y_test_ridge = y_dict.get(max_cv_score_ridge[1])

    best_x_test_lasso = x_dict.get(max_cv_score_lasso[1])
    best_y_test_lasso = y_dict.get(max_cv_score_lasso[1])

    # Best fit model scores to be used as our regularization complexities, AKA alpha
    max_score_ridge = max_cv_score_ridge[0]
    max_score_lasso = max_cv_score_lasso[0]

    # Task 1
    print("Best data")
    print('\nOrdinary Least Squares Regression')
    reg_suite.ols_regression(max_score_x_data, max_score_y_data, best_x_test, best_y_test)
    print('\nRidge Regression')
    reg_suite.ridge_regression(max_score_x_data_ridge, max_score_y_data_ridge,
                               best_x_test_ridge, best_y_test_ridge, max_score_ridge)
    print('\nLasso Regression')
    reg_suite.lasso_regression(max_score_x_data_lasso, max_score_y_data_lasso,
                               best_x_test_lasso, best_y_test_lasso, max_score_lasso)


slump_flow()
