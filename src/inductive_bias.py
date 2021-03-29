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

class IB(DL):
    def __init__(self):
        '''
        ['Gaussian XOR', 'Uniform XOR', 'Spiral', 'Gaussian R-XOR', 'Gaussian S-XOR']
        '''
        self.date = datetime.now()
        self.spiral = DG.spiral_center(N=270, rng=1)

        self.truepst = [[[] for i in range(5)] for j in range(2)]

        self.train_X = [[] for i in range(5)]
        self.train_y = [[] for i in range(5)]
        self.test_X = [[] for i in range(5)]
        self.test_y = [[] for i in range(5)]

        self.Ctrain_X = [[] for i in range(5)]
        self.Ctrain_y = [[] for i in range(5)]
        self.Ctest_X = [[] for i in range(5)]
        self.Ctest_y = [[] for i in range(5)]

    def get_posterior(self, **kwargs):
        '''
        generating preset true posterior distribution for the datasets
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
                    print(arg)
                    self.truepst[i][j] = DG.true_spiral(**arg)
                elif j == 3:
                    arg['rotate'] = True
                    self.truepst[i][j] = DG.true_xor(**arg)
                elif j == 4:
                    arg['sig'] = 0.1
                    self.truepst[i][j] = DG.true_xor(**arg)

    def get_trueSpiral(self, l, r, h, sig=0.00008, rng=4.3, cc=True, spirals=270):
        '''
        get true posterior for spiral dataset
        '''
        self.truepst = DG.true_spiral(l=l, r=r, h=h, sig=sig, rng=rng, cc=cc, spirals=spirals)

    def get_testpdfSpiral(self, N, K=2, noise=1, acorn=None, density=0.5, rng=1):
        '''
        get exploratory true posterior for spiral dataset (deprecated and replaced by get_trueSpiral)
        '''
        self.truepst = DG.pdf_spiral(N, K, noise, acorn, density, rng)

    def get_trueXOR(self, l=-2, r=2, h=0.01, rotate=False, sig=0.25, cc=False):
        '''
        get true posterior for XOR dataset
        '''
        self.truepst = DG.true_xor(l, r, h, rotate, sig, cc)

    def get_trueUXOR(self, l=-2, r=2, h=0.01, cc=False):
        '''
        get true posterior for uniform XOR dataset
        '''
        self.truepst = DG.true_Uxor(l, r, h, cc)

    def get_colors(colors, inds):
        '''
        get colors for two classes
        '''
        return [colors[i] for i in inds]

    def train_models():
        pass

