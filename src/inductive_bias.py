'''
Base class for inductive bias experiment

Author: Jong M. Shin
'''

import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
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

        self.humanLoc = [] #MTurk human coordinates

        self.spiral = DG.spiral_center(N=270, rng=3)

        self.truepst = [[[] for i in range(d)] for j in range(2)]
        self.estpst = [[[] for i in range(d)] for j in range(2)]
        self.clf = [[[] for i in range(d)] for j in range(2)]
        self.hdist = None
        self.human = None
        seed = np.array([[],[],[]]).T
        self.estpst_sample = [[seed for i in range(m)] for i in range(2)] #only for spiral and S-XOR
        self.hdist_sample = [[seed for i in range(m)] for i in range(2)] #only for spiral and S-XOR

        self.mask = DG.generate_mask(rng=4.3)
        # self.mask = DG.generate_mask(rng=3, h=0.1)

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

    def get_dataset(self, N=100, cov=1, rng=1):
        '''
        Generates simulation datasets

        N: number of samples
        cov: covariance
        rng: range

        index = ['Gaussian XOR', 'Uniform XOR', 'Spiral', 'Gaussian R-XOR', 'Gaussian S-XOR']
        '''

        mean = np.array([-rng, -rng])

        for i in range(5):

            if i == 0:  # gaussian XOR
                args = {'n': N, 'mean': mean,
                        'cov_scale': cov/10, 'angle_params': np.pi}
            elif i == 3:  # gaussian R-XOR
                args = {'n': N, 'mean': mean, 'cov_scale': cov /
                        10, 'angle_params': np.pi/4}
            elif i == 4:  # gaussian S-XOR
                args = {'n': N, 'mean': mean,
                        'cov_scale': cov/100, 'angle_params': np.pi}
            elif i == 1:  # gaussian U-XOR
                args = {'N': N, 'b': rng}
            elif i == 2:  # gaussian Spiral
                args = {'N': N, 'K': 2, 'noise': 1, 'density': 0.3, 'rng': rng}

            if i == 1:
                func = DG.generate_uniform_XOR
            elif i == 2:
                func = DG.generate_spirals
            else:
                func = DG.generate_gaussian_parity

            self.train_X[i], self.train_y[i] = func(**args)
            self.test_X[i], self.test_y[i] = func(**args)
            self.Ctrain_X[i], self.Ctrain_y[i] = func(**args, cc=True)
            self.Ctest_X[i], self.Ctest_y[i] = func(**args, cc=True)

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

            for j in range(len(self.dtype)):

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

    def get_clf(self, param=None, fast=False):
        '''
        train ML models with the simulation datasets

        param: preset hyper-parameters for fast train. If fast=True, this argument is required
        fast: if True, trains on pre-defined hyper-parameters without grid searching
        '''
        self.clf = [[[] for i in range(len(self.dtype))]
                    for j in range(2)]  # reinitialize

        for j in tqdm(range(2), leave=False, desc='train clf'):
            if j == 0: continue #skipping unit square
            for i in range(len(self.dtype)):
                
                if j == 0:
                    args = dict(param=param[j][i], dset=i, enable=[0, 0, 1, 0, 1, 1], cc=False)
                else:
                    args = dict(param=param[j][i], dset=i, enable=[0, 0, 1, 0, 1, 1], cc=True)

                if fast:
                    self.clf[j][i] = self.fast_train(**args)
                else:
                    self.clf[j][i] = self.train(**args)

    def get_proba(self, human_idx=None):
        '''
        get estimated posterior probability

        exact[bool]: predicts exact locations used for MTurk human experiment
        '''
        self.estpst = [[[] for i in range(len(self.dtype))]
                       for j in range(2)]  # reinitialize

        dset_limiter = np.array([2,4]) #limits only spiral and sxor

        for i in tqdm(range(len(self.clf)), leave=False, desc='predict_proba'):  # either square or circular boundary
            if i == 0: continue
            for j, cl in enumerate(self.clf[i]):  # datasets
                if j in dset_limiter:
                    for md in cl:
                        temp = None  # for some reason, this is required for RXOR
                        if human_idx:
                            temp_idx = np.argmax(dset_limiter == j)
                            temp = md.predict_proba(self.humanLoc[temp_idx][human_idx].to_numpy())
                        else:
                            temp = md.predict_proba(self.mask)
                        self.estpst[i][j].append(temp[:, 1]) #make sure to choose the right probability corresponding to the true posterior

    def get_hellinger(self, fast=False, rep=None):
        '''
        get hellinger distance between estimated and true posterior distributions

        output
        ------
        hdist: computed hellinger distance in i x j x k array 
        where i is a type of unit boundary (square: 0, circle: 1), 
        j is a type of datasets, and k is a type of ML models
        '''
        if rep: #exact coordinate method not implemented yet
            pass
        else:
            self.hdist = self.compute_hellinger(
            estP=self.estpst, trueP=self.truepst, fast=fast)

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

    def get_sampledData(self, saved_clf, reps, N_sample, **kwargs):
        '''
        simulate human behavioral experiment setting on ML experiment by sampling 
        'N_sample' number of points from each estimation for 'reps' number of repetition

        saved_clf: load previously saved hyper-parameters of classifer (currently only for spiral and S-XOR)
        reps: number of independent experiment
        N_sample: number of sampled points 

        output
        ------
        estpst_sample: N x 3 matrix in I X J multi-dimensional list where first two columns of the matrix are x,y coordinates and the third human estimates
        hdist_sample: N x 3 matrix in I X J multi-dimensional list where first two columns of the matrix are x,y coordinates and the third human hellinger distance

        ** the method automatically overwrites previous pickle file
        '''

        seed = np.array([[],[],[]]).T
        self.estpst_sample = [[seed for i in range(len(self.mtype))] for i in range(2)] #only for spiral and S-XOR

        self.get_posterior(**kwargs) #re-generate true posterior

        for rep in tqdm(range(reps), desc='rep'):

            self.get_dataset() #sampling at N=100 by default
            self.get_clf(param=saved_clf, fast=True)
            self.get_proba(rep) #if rep, exact coord match with human
            self.get_hellinger(fast=True, rep=rep)
            
            # append estimate posterior and hellinger distance
            for j_i, j in enumerate([2,4]):
                for i in range(len(self.mtype)):
                    if rep:
                        dat_temp = np.column_stack([self.humanLoc[j_i][rep], self.estpst[1][j][i]]) #select only the circular boundary
                        self.estpst_sample[j_i][i] = np.vstack([self.estpst_sample[j_i][i],dat_temp])
                    else:
                        dat_temp = np.column_stack([self.mask, self.estpst[1][j][i]]) #select only the circular boundary
                        self.estpst_sample[j_i][i] = self.sample(self.estpst_sample[j_i][i], dat_temp, N_sample)

                    if rep: #exact coordinate method not implemented yet
                        pass
                    else:
                        dat_temp = np.column_stack([self.mask, self.hdist[1][j][i]]) #select only the circular boundary
                        self.hdist_sample[j_i][i] = self.sample(self.hdist_sample[j_i][i], dat_temp, N_sample)

        self.load_sampledData(save=True)

    def get_linegrid(self):
        '''
        '''
        return None

    def _loadall(self):
        '''
        convenient method that loads datasets, trained classifiers, true posteriors, 
        estimated posteriors, hellinger distance between true and estimated posterior,
        MTurk human data
        '''
        self.load_dataset()
        self.load_posterior()
        self.load_clf()
        self.load_est()
        self.load_hellinger()
        # self.load_sampledData() #too much memory, switched to manual load
        self.load_MTurk(verbose=True)

    def get_colors(colors, inds):
        '''
        get colors for two classes
        '''
        return [colors[i] for i in inds]
