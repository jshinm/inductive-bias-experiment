'''
Subclass dataset loader for inductive bias experiment

Author: Jong M. Shin
'''

import os
import pickle
import numpy as np
from datetime import datetime

class datasetLoader:

    def generate(self, N=100, cov=1, rng=1):

        mean = np.array([-rng,-rng])

        for i in range(5):

            if i == 0: # gaussian XOR
                args = {'n': N, 'mean': mean, 'cov_scale': cov/10, 'angle_params': np.pi}
            elif i == 1: # gaussian R-XOR
                args = {'n': N, 'mean': mean, 'cov_scale': cov/10, 'angle_params': np.pi/4}
            elif i == 2: # gaussian S-XOR
                args = {'n': N, 'mean': mean, 'cov_scale': cov/100, 'angle_params': np.pi}
            elif i == 3: # gaussian U-XOR
                args = {'N': N, 'b': rng}
            elif i == 4: # gaussian Spiral
                args = {'N': N, 'K': 2, 'noise': 1, 'density': 0.3, 'rng': rng}

            if i == 3:
                func = DG.generate_uniform_XOR
            elif i == 4:
                func = DG.generate_spirals
            else:
                func = DG.generate_gaussian_parity

            self.train_X[i], self.train_y[i] = func(**args)
            self.test_X[i], self.test_y[i] = func(**args)
            self.Ctrain_X[i], self.Ctrain_y[i] = func(**args, cc=True)
            self.Ctest_X[i], self.Ctest_y[i] = func(**args, cc=True)
                
    def load(self, fname='SimulationData.pickle', save=False):
        '''
        loads saved simulation dataset or saves current attributes as a pickle
        '''
        CLFPATH = os.path.join(os.getcwd(), 'clf\\')
        filename = CLFPATH + fname

        if os.path.exists(filename) and not save:
            with open(filename, 'rb') as f:
                dataset = pickle.load(f, encoding='bytes')

            for j in range(2): #square vs circular boundary
                for i in range(5):
                    if j == 0:
                        self.train_X[i], self.train_y[i], self.test_X[i], self.test_y[i] = dataset[j][i]
                    elif j == 1:
                        self.Ctrain_X[i], self.Ctrain_y[i], self.Ctest_X[i], self.Ctest_y[i] = dataset[j][i]

            print('[', filename, '] loaded')

        else: 
            print('creating new datasets..')
            sTime = datetime.now()
            
            if not os.path.isdir(CLFPATH):
                os.makedirs(CLFPATH)

            with open(filename, 'wb') as f:
                temp = []
                for j in range(2):
                    temp.append([])
                    for i in range(5):
                        if j == 0:
                            temp[j].append([self.train_X[i], self.train_y[i], self.test_X[i], self.test_y[i]])
                        elif j == 1:
                            temp[j].append([self.Ctrain_X[i], self.Ctrain_y[i], self.Ctest_X[i], self.Ctest_y[i]])
                pickle.dump(temp, f)

            deltaT = datetime.now() - sTime
            print('completed after ' + str(deltaT.seconds) + ' seconds')
            print('saved as [', filename, ']')

    def read(self, dset):
        '''
        read train and test dataset
        '''
        return train_X[dset], train_y[dset], test_X[dset], test_y[dset]