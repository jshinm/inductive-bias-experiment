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

class IB(DL, TM):
    def __init__(self):
        '''
        ['Gaussian XOR', 'Uniform XOR', 'Spiral', 'Gaussian R-XOR', 'Gaussian S-XOR']
        '''
        self.date = datetime.now()
        self.mtype = ['SVM', 'MLP', 'RF', 'QDA'] # list of ML models (SVC removed on 12/8/2020), (KNN/XGBoost removed on 12/9/2020)
        self.dtype = ['Gaussian XOR', 'Uniform XOR', 'Spiral', 'Gaussian R-XOR', 'Gaussian S-XOR'] # list of datasets

        m = len(self.mtype)
        d = len(self.dtype)

        self.spiral = DG.spiral_center(N=270, rng=3)

        self.truepst = [[[] for i in range(d)] for j in range(2)]
        self.clf = [[[] for i in range(m)] for j in range(2)]

        self.train_X = [[] for i in range(d)]
        self.train_y = [[] for i in range(d)]
        self.test_X = [[] for i in range(d)]
        self.test_y = [[] for i in range(d)]

        self.Ctrain_X = [[] for i in range(d)]
        self.Ctrain_y = [[] for i in range(d)]
        self.Ctest_X = [[] for i in range(d)]
        self.Ctest_y = [[] for i in range(d)]

    def get_spiralCenter(self, **kwargs):
        '''
        generate spiral centers used to populate gaussian posterior

        N: number of spiral centers
        K: number of spiral (K=2)
        noise: gaussian noise
        density: spiral center density
        rng: range
        '''

        self.spiral = DG.generate_spirals(**kwargs)#.spiral_center(**kwargs)

    def get_posterior(self, **kwargs):
        '''
        generating preset true posterior distribution for the datasets

        h: grid density
        sig: covariance
        rng: range
        spirals: number of seed spirals (each spiral generates gaussian posterior)
        '''

        for i in range(2):

            for j in range(5):

                arg = kwargs.copy() #reinitialize args

                if i == 0:
                    arg['cc'] = False
                else:
                    arg['cc'] = True

                if j == 0:
                    self.truepst[i][j] = DG.true_xor(**arg)
                elif j == 1:                    
                    self.truepst[i][j] = DG.true_Uxor(**arg)
                elif j == 2:
                    arg['rng'] = 4.3
                    arg['sig'] = 0.00008
                    self.truepst[i][j] = DG.true_spiral(**arg)
                elif j == 3:
                    arg['rotate'] = True
                    self.truepst[i][j] = DG.true_xor(**arg)
                elif j == 4:
                    arg['sig'] = 0.1
                    self.truepst[i][j] = DG.true_xor(**arg)

    def load_posterior(self, fname='PosteriorData.pickle', save=False):
        '''
        loads saved posterior distribution for all datasets or saves current attributes as a pickle
        '''
        CLFPATH = os.path.join(os.getcwd(), 'clf\\')
        filename = CLFPATH + fname

        if os.path.exists(filename) and not save:
            with open(filename, 'rb') as f:
                self.truepst = pickle.load(f, encoding='bytes')

            print('[', filename, '] loaded')

        else: 
            print('saving current posterior..')
            sTime = datetime.now()
            
            if not os.path.isdir(CLFPATH):
                os.makedirs(CLFPATH)

            with open(filename, 'wb') as f:
                pickle.dump(self.truepst, f)

            deltaT = datetime.now() - sTime
            print('completed after ' + str(deltaT.seconds) + ' seconds')
            print('saved as [', filename, ']')      

    def get_clf(self):
        '''
        '''        
        for j in range(2):
            for i in range(len(self.mtype)):
                if j == 0:
                    args = dict(dset=i, enable=[0,0,1,0,1,1], cc=False)
                else:
                    args = dict(dset=i, enable=[0,0,1,0,1,1], cc=True)

                self.clf[j][i] = TM.train_model(**args)




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

    def train_models():
        pass

