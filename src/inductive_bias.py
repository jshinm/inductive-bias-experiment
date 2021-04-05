'''
Base class for inductive bias experiment

Author: Jong M. Shin
'''

import os
import pickle
import numpy as np
from datetime import datetime
from .dataset_generator import DatasetGenerator as DG
from .dataset_loader import datasetLoader as DL
from .train_model import trainModel as TM
from .model_analysis import modelAnalysis as MA


class IB(DL, TM, MA):
    def __init__(self):

        self.date = datetime.now()
        # list of ML models (SVC removed on 12/8/2020), (KNN/XGBoost removed on 12/9/2020)
        self.mtype = ['SVM', 'MLP', 'RF', 'QDA']
        self.dtype = ['Gaussian XOR', 'Uniform XOR', 'Spiral',
                      'Gaussian R-XOR', 'Gaussian S-XOR']  # list of datasets

        m = len(self.mtype)
        d = len(self.dtype)

        self.spiral = DG.spiral_center(N=270, rng=3)

        self.truepst = [[[] for i in range(d)] for j in range(2)]
        self.estpst = [[[] for i in range(d)] for j in range(2)]
        self.clf = [[[] for i in range(d)] for j in range(2)]
        self.hdist = None
        self.human = None

        self.mask = DG.generate_mask(rng=4.3)

        self.train_X = [[] for i in range(d)]
        self.train_y = [[] for i in range(d)]
        self.test_X = [[] for i in range(d)]
        self.test_y = [[] for i in range(d)]

        self.Ctrain_X = [[] for i in range(d)]
        self.Ctrain_y = [[] for i in range(d)]
        self.Ctest_X = [[] for i in range(d)]
        self.Ctest_y = [[] for i in range(d)]

        self._loadall()

    def get_spiralCenter(self, **kwargs):
        '''
        generate spiral centers used to populate gaussian posterior

        N: number of spiral centers
        K: number of spiral (K=2)
        noise: gaussian noise
        density: spiral center density
        rng: range
        '''
        self.spiral = DG.spiral_center(**kwargs)
        # self.spiral = DG.generate_spirals(**kwargs)  # .spiral_center(**kwargs)

    def get_posterior(self, **kwargs):
        '''
        generating preset true posterior distribution for the datasets

        h: grid density
        sig: covariance
        rng: range
        spirals: number of seed spirals (each spiral generates gaussian posterior)

        output
        ------
        x: a vector of x-coordinates of the grid 
        y: a vector of y-coordinates of the grid
        posterior: a vector of posterior probability
        '''
        self.truepst = [[[] for i in range(len(self.dtype))] for j in range(2)]

        for i in range(2):

            for j in range(self.dtype):

                arg = kwargs.copy()  # reinitialize args

                if i == 0:
                    arg['cc'] = False
                else:
                    arg['cc'] = True

                if j == 0:
                    self.truepst[i][j] = DG.true_xor(**arg)
                elif j == 1:
                    self.truepst[i][j] = DG.true_Uxor(**arg)
                elif j == 2:
                    arg['sig'] = 0.00008
                    self.truepst[i][j] = DG.true_spiral(**arg)
                elif j == 3:
                    arg['rotate'] = True
                    self.truepst[i][j] = DG.true_xor(**arg)
                elif j == 4:
                    arg['sig'] = 0.1
                    self.truepst[i][j] = DG.true_xor(**arg)

    def get_clf(self):
        '''
        train ML models with the simulation datasets
        '''
        self.clf = [[[] for i in range(len(self.dtype))]
                    for j in range(2)]  # reinitialize

        for j in range(2):
            for i in range(len(self.dtype)):
                if j == 0:
                    args = dict(dset=i, enable=[0, 0, 1, 0, 1, 1], cc=False)
                else:
                    args = dict(dset=i, enable=[0, 0, 1, 0, 1, 1], cc=True)

                self.clf[j][i] = self.train(**args)

    def get_proba(self):
        '''
        get estimated posterior probability
        '''
        self.estpst = [[[] for i in range(len(self.dtype))]
                       for j in range(2)]  # reinitialize

        for i in range(len(self.clf)):  # either square or circular boundary
            for j, cl in enumerate(self.clf[i]):  # datasets
                for md in cl:
                    temp = None  # for some reason, this is required for RXOR
                    temp = md.predict_proba(self.mask)
                    self.estpst[i][j].append(temp[:, 0])

    def get_hellinger(self):
        '''
        get hellinger distance between estimated and true posterior distributions

        output
        ------
        hdist: computed hellinger distance in i x j x k array 
        where i is a type of unit boundary (square: 0, circle: 1), 
        j is a type of datasets, and k is a type of ML models
        '''

        self.hdist = self.compute_hellinger(
            estP=self.estpst, trueP=self.truepst)

    def get_radialDist(self, dat):
        '''
        get smoothed radial distance

        dat: a N x 3 matrix where there are N number of samples
        and three columns of x, y, variable of interest
        origin: origin of the distance
        step: step in which the search radius is increasing
        end: outer boundary of the search
        srng: smoothing range (aka width of the search ring)

        output
        ------
        returns a 2-D list of radial distance and vectors of variable of interest
        '''
        return self.smooth_radial_distance(dat)

    def _loadall(self):
        '''
        convenient method that loads datasets, trained classifiers, true posteriors, 
        estimated posteriors, hellinger distance between true and estimated posterior
        '''
        self.load_dataset()
        self.load_posterior()
        self.load_clf()
        self.load_est()
        self.load_hellinger()
        self.load_MTurk()

    def get_testpdfSpiral(self, N, K=2, noise=1, acorn=None, density=0.5, rng=1):
        '''
        get exploratory true posterior for spiral dataset (deprecated and replaced by get_trueSpiral)
        '''
        self.truepst = DG.pdf_spiral(N, K, noise, acorn, density, rng)

    def get_colors(colors, inds):
        '''
        get colors for two classes
        '''
        return [colors[i] for i in inds]
