'''
Subclass ML training module for inductive bias experiment

Author: Jong M. Shin
'''

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import xgboost as xgb


class trainModel:

    def train(self, dset, enable=[0, 0, 1, 0, 1, 1], cc=False):

        mods = ['KNN', 'SVC', 'SVM', 'XGBoost', 'MLP', 'RF']  # local mods list

        def tune_param(clf, param, name):

            model = GridSearchCV(clf, param_grid=param,
                                 return_train_score=True, cv=20)
            model.fit(train_X, train_y)

            print(name + " INFO:")
            print("Best hyper paramters:", model.best_params_)
            print("Best accuracy value: ", model.best_score_)

            clf.set_params(**model.best_params_)
            clf.fit(train_X, train_y)  # actually fitting the model
            print("prediction score: ", model.score(test_X, test_y))
            print(clf)

            # plot_posterior(X, y, newX, newy, clf, name, savefile)
            # xx, yy, Z, new_p, zz = plot_decision_boundaries(X, y ,clf, h=h)
            return clf

        post = []

        if cc:
            train_X, train_y, test_X, test_y = self.Ctrain_X[
                dset], self.Ctrain_y[dset], self.Ctest_X[dset], self.Ctest_y[dset]
        else:
            train_X, train_y, test_X, test_y = self.train_X[
                dset], self.train_y[dset], self.test_X[dset], self.test_y[dset]

        ##### KNN #####

        if enable[0] == 1:
            tuned_param = [{'n_neighbors': [3, 5, 7], 'leaf_size':range(10, 100, 10)},
                           {'n_neighbors': [3, 5, 7],
                               'leaf_size':range(10, 100, 10)},
                           {'n_neighbors': [7]},
                           {'n_neighbors': [3, 5, 7],
                               'leaf_size':range(10, 100, 10)},
                           {'n_neighbors': [3, 5, 7], 'leaf_size':range(10, 100, 10)}]
            temp = tune_param(KNeighborsClassifier(),
                              tuned_param[dset], mods[0])
            post.append(temp)

        ##### SVC #####

        if enable[1] == 1:
            tuned_param = [{'gamma': ['auto'], 'probability':[True]},
                           {'gamma': ['auto'], 'probability':[True]},
                           {'C': np.linspace(1, 10, 10),
                            'probability': [True]},
                           {'gamma': ['auto'], 'probability':[True]},
                           {'gamma': ['auto'], 'probability':[True]}]
            temp = tune_param(svm.SVC(), tuned_param[dset], mods[1])
            post.append(temp)

        ##### nuSVC #####

        if enable[2] == 1:
            tuned_param = [{'gamma': ['auto'], 'probability':[True]},
                           {'gamma': ['auto'], 'probability':[True]},
                           {'probability': [True]},
                           {'gamma': ['auto'], 'probability':[True]},
                           {'gamma': ['auto'], 'probability':[True]}]
            temp = tune_param(svm.NuSVC(), tuned_param[dset], mods[2])
            post.append(temp)

        ##### xgbooster #####

        if enable[3] == 1:
            tuned_param = [{'n_jobs': [-1], 'learning_rate':np.linspace(0, 1, 20), 'n_estimators':[64, 128, 256],
                            'gamma': [0], 'objective':['binary:logistic']},
                           {'n_jobs': [-1], 'learning_rate':np.linspace(0, 1, 20), 'n_estimators':[64, 128, 256],
                            'gamma': [0], 'objective':['binary:logistic']},
                           {'n_jobs': [-1], 'learning_rate':np.linspace(0, 1, 20), 'n_estimators':[64, 128, 256],
                            'gamma': [0]},
                           {'n_jobs': [-1], 'learning_rate':np.linspace(0, 1, 20), 'n_estimators':[64, 128, 256],
                            'gamma': [0], 'objective':['binary:logistic']},
                           {'n_jobs': [-1], 'learning_rate':np.linspace(0, 1, 20), 'n_estimators':[64, 128, 256],
                            'gamma': [0], 'objective':['binary:logistic']}]
            temp = tune_param(xgb.XGBClassifier(objective='binary:logistic'),
                              tuned_param[dset], mods[3])
            post.append(temp)

        #### mlp #####

        if enable[4] == 1:
            tuned_param = [{'alpha': [0], 'max_iter':[7000], 'hidden_layer_sizes':[100], 'learning_rate_init':[0.0001]},
                           {'alpha': [0], 'max_iter':[7000], 'hidden_layer_sizes':[
                               100], 'learning_rate_init':[0.0001]},
                           {'alpha': [0], 'max_iter':[10000], 'activation':[
                               'logistic', 'relu'], 'learning_rate_init':[0.0001], 'solver': ['lbfgs']},
                           {'alpha': [0], 'max_iter':[7000], 'hidden_layer_sizes':[
                               100], 'learning_rate_init':[0.0001]},
                           {'alpha': [0], 'max_iter':[7000], 'hidden_layer_sizes':[100], 'learning_rate_init':[0.0001]}]
            temp = tune_param(MLPClassifier(), tuned_param[dset], mods[4])
            post.append(temp)

        #### RF #####

        if enable[5] == 1:
            tuned_param = [{'max_depth': [10], 'n_estimators':[128]},
                           {'max_depth': [10], 'n_estimators':[128]},
                           {'max_depth': [10], 'n_estimators':[128]},
                           {'max_depth': [10], 'n_estimators':[128]},
                           {'max_depth': [10], 'n_estimators':[128]}]
            temp = tune_param(RandomForestClassifier(n_jobs=-1),
                              tuned_param[dset], mods[5])
            post.append(temp)

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(train_X, train_y)
        post.append(qda)

        return post
        # plot_posterior(X, y, newX, newy, post, mods, savefile, h=h)

    def train_store_algo(filename, n=0, cc=False):
        '''
        Trains and stores classifiers

        n: dataset being trained on (0-4)
        '''
        if path.exists(filename):
            with open(filename, 'rb') as f:
                clf = pickle.load(f, encoding='bytes')
            print('[', filename, '] loaded')

            return clf

        else:
            print('begin training..')
            sTime = datetime.now()

            clf = train_algo(n, cc=cc)

            if not os.path.isdir(CLFPATH):
                os.makedirs(CLFPATH)

            with open(filename, 'wb') as f:
                pickle.dump(clf, f)

            deltaT = datetime.now() - sTime
            print('completed after ' + str(deltaT.seconds) + ' seconds')
            print('saved as [', filename, ']')

            return clf
