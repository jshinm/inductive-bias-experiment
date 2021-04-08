'''
Subclass dataset loader for inductive bias experiment

Author: Jong M. Shin
'''

import os
import pickle
import numpy as np
from datetime import datetime
from .dataset_generator import DatasetGenerator as DG


class datasetLoader:

    # def generate_dataset(self, N=100, cov=1, rng=1):
    #     '''
    #     Generates simulation datasets

    #     N: number of samples
    #     cov: covariance
    #     rng: range

    #     index = ['Gaussian XOR', 'Uniform XOR', 'Spiral', 'Gaussian R-XOR', 'Gaussian S-XOR']
    #     '''

    #     mean = np.array([-rng, -rng])

    #     for i in range(5):

    #         if i == 0:  # gaussian XOR
    #             args = {'n': N, 'mean': mean,
    #                     'cov_scale': cov/10, 'angle_params': np.pi}
    #         elif i == 3:  # gaussian R-XOR
    #             args = {'n': N, 'mean': mean, 'cov_scale': cov /
    #                     10, 'angle_params': np.pi/4}
    #         elif i == 4:  # gaussian S-XOR
    #             args = {'n': N, 'mean': mean,
    #                     'cov_scale': cov/100, 'angle_params': np.pi}
    #         elif i == 1:  # gaussian U-XOR
    #             args = {'N': N, 'b': rng}
    #         elif i == 2:  # gaussian Spiral
    #             args = {'N': N, 'K': 2, 'noise': 1, 'density': 0.3, 'rng': rng}

    #         if i == 1:
    #             func = DG.generate_uniform_XOR
    #         elif i == 2:
    #             func = DG.generate_spirals
    #         else:
    #             func = DG.generate_gaussian_parity

    #         self.train_X[i], self.train_y[i] = func(**args)
    #         self.test_X[i], self.test_y[i] = func(**args)
    #         self.Ctrain_X[i], self.Ctrain_y[i] = func(**args, cc=True)
    #         self.Ctest_X[i], self.Ctest_y[i] = func(**args, cc=True)

    def load_dataset(self, fname='SimulationData.pickle', save=False):
        '''
        loads saved simulation dataset or saves current attributes as a pickle
        '''
        CLFPATH = os.path.join(os.getcwd(), 'clf\\')
        filename = CLFPATH + fname

        if os.path.exists(filename) and not save:
            with open(filename, 'rb') as f:
                dataset = pickle.load(f, encoding='bytes')

            for j in range(2):  # square vs circular boundary
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
                            temp[j].append(
                                [self.train_X[i], self.train_y[i], self.test_X[i], self.test_y[i]])
                        elif j == 1:
                            temp[j].append(
                                [self.Ctrain_X[i], self.Ctrain_y[i], self.Ctest_X[i], self.Ctest_y[i]])
                pickle.dump(temp, f)

            deltaT = datetime.now() - sTime
            print('completed after ' + str(deltaT.seconds) + ' seconds')
            print('saved as [', filename, ']')

    def _load(self, fname, target, save=False):
        '''
        loads previously saved attributes from a pickle or saves current attributes as a pickle
        '''
        CLFPATH = os.path.join(os.getcwd(), 'clf\\')
        filename = CLFPATH + fname

        if os.path.exists(filename) and not save:
            with open(filename, 'rb') as f:
                target = pickle.load(f, encoding='bytes')

            print('[', filename, '] loaded')

        else:
            print('saving current attributes..')
            sTime = datetime.now()

            if not os.path.isdir(CLFPATH):
                os.makedirs(CLFPATH)

            with open(filename, 'wb') as f:
                pickle.dump(target, f)

            deltaT = datetime.now() - sTime
            print('completed after ' + str(deltaT.seconds) + ' seconds')
            print('saved as [', filename, ']')

        return target

    def load_posterior(self, fname='PosteriorData.pickle', save=False):
        '''
        loads saved posterior distribution for all datasets or saves current attributes as a pickle
        '''
        self.truepst = self._load(
            fname='PosteriorData.pickle', target=self.truepst, save=save)

    def load_clf(self, fname='TrainedCLF.pickle', save=False):
        '''
        loads saved classifiers trained on all datasets or saves current classifier attributes as a pickle
        '''
        self.clf = self._load(fname='TrainedCLF.pickle',
                              target=self.clf, save=save)

    def load_est(self, save=False):
        '''
        loads posterior prediction from the trained sklearn classifiers using predict_proba() or saves current posterior attributes as a pickle
        '''
        self.estpst = self._load(
            fname='EstimatedData.pickle', target=self.estpst, save=save)

    def load_hellinger(self, save=False):
        '''
        loads previously computed hellinger distance or saves current hellinger attributes as a pickle
        '''
        self.hdist = self._load(fname='HellingerData.pickle',
                                target=self.hdist, save=save)

    def load_sampledData(self, save=False):
        '''
        loads previously computed sampled data or saves current hellinger attributes as a pickle
        '''
        self.estpst_sample, self.hdist_sample = self._load(fname='SampledData.pickle',
                                target=[self.estpst_sample, self.hdist_sample], save=save)

    def load_MTurk(self, verbose=False):
        '''
        load MTurk human behavioral experiment data

        D_{j=0}: human estimate
        D_{j=1}: true posterior
        D_{j=2}: dataset type (0: S-XOR, 1: Spiral)
        D_{j=3}: x-coordinate
        D_{j=4}: euclidean distance from the origin (0,0)
        D_{j=5}: y-coordinate
        D_{j=6}: hellinger distance between human estimate and true posterior
        D_{j=7}: participant ID
        '''

        loadall = False  # switch to include both actual experiment and pilot data

        # load cleaned MTurk data
        with open('dat/MTurk_ds.pickle', 'rb') as f:
            MT_ds = pickle.load(f)

        # loads pilot data in addition to the actual data
        if loadall:
            with open('dat/MTurk_ds_pilot.pickle', 'rb') as f:
                MT_ds_pilot = pickle.load(f)
            MT_ds = np.vstack((MT_ds, MT_ds_pilot))

        # select by dataset ('est', 'real', 'mtype', 'x', 'd', 'y')
        MT_sxor = MT_ds[MT_ds[:, 2] == 2]  # S-XOR
        MT_spir = MT_ds[MT_ds[:, 2] == 4]  # Spiral

        if verbose:
            print(
                f'Size of the S-XOR: {MT_sxor[:,3].shape}\
                    \nSize of the Spiral: {MT_spir[:,3].shape}\
                    \nSize of the whole dataset: {MT_ds.shape}')

        # compute hellinger distance between est and real
        temp_d = self._hellinger_explicit(MT_sxor[:, 0], MT_sxor[:, 1])
        MT_sxor = np.column_stack((MT_sxor, temp_d))
        MT_sxor[:, [6, 7]] = MT_sxor[:, [7, 6]]
        # temp_d = abs(MT_4[:,0] - MT_4[:,1])
        temp_d = self._hellinger_explicit(MT_spir[:, 0], MT_spir[:, 1])
        MT_spir = np.column_stack((MT_spir, temp_d))
        MT_spir[:, [6, 7]] = MT_spir[:, [7, 6]]

        if verbose:
            print(f'\nSize of the S-XOR after adding hellinger: {MT_sxor.shape}\
            \nSize of the Spiral after adding hellinger: {MT_spir.shape}')  # 'est', 'real', 'mtype', 'x', 'd', 'y', 'hellinger', 'id'

        self.human = [MT_spir, MT_sxor]
