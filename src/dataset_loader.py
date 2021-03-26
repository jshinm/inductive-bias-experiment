import numpy as np
import os
import pickle
from datetime import datetime
from .dataset_generator import DatasetGenerator as DG

class datasetLoader:

    def __init__(self):
        # self.N = N #training sample size
        # self.train_rng = train_rng #training set range
        # self.test_rng  = test_rng #testing set range

        
        # test_mean = np.array([-test_rng,-test_rng])

        self.train_X = [[] for i in range(5)]
        self.train_y = [[] for i in range(5)]
        self.test_X = [[] for i in range(5)]
        self.test_y = [[] for i in range(5)]

        self.Ctrain_X = [[] for i in range(5)]
        self.Ctrain_y = [[] for i in range(5)]
        self.Ctest_X = [[] for i in range(5)]
        self.Ctest_y = [[] for i in range(5)]

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
                
        # # gaussian XOR
        # i = 0
        
        # self.train_X[i], self.train_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi)
        # self.test_X[i], self.test_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi, train=False)
        # self.Ctrain_X[i], self.Ctrain_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi, cc=True)
        # self.Ctest_X[i], self.Ctest_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi, cc=True, train=False)

        # # gaussian R-XOR
        # i = 1
        # self.train_X[i], self.train_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi/4)
        # self.test_X[i], self.test_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi/4, train=False)
        # self.Ctrain_X[i], self.Ctrain_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi/4, cc=True)
        # self.Ctest_X[i], self.Ctest_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/10, angle_params=np.pi/4, cc=True, train=False)

        # # gaussian S-XOR
        # i = 2
        # self.train_X[i], self.train_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/100, angle_params=np.pi)
        # self.test_X[i], self.test_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/100, angle_params=np.pi, train=False)
        # self.Ctrain_X[i], self.Ctrain_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/100, angle_params=np.pi, cc=True)
        # self.Ctest_X[i], self.Ctest_y[i] = DG.generate_gaussian_parity(n=N, mean=mean, cov_scale=cov/100, angle_params=np.pi, cc=True, train=False)
        
        # # gaussian U-XOR
        # i = 3
        # self.train_X[i], self.train_y[i] = DG.generate_uniform_XOR(N=N, b=rng)
        # self.test_X[i], self.test_y[i] = DG.generate_uniform_XOR(N=N, b=rng)
        # self.Ctrain_X[i], self.Ctrain_y[i] = DG.generate_uniform_XOR(N=N, b=rng, cc=True)
        # self.Ctest_X[i], self.Ctest_y[i] = DG.generate_uniform_XOR(N=N, b=rng, cc=True)

        # # gaussian Spiral
        # i = 4 #X, Y = generate.generate_spirals(n, 2, noise=1, rng=1, density=0.3) from behavioral
        # self.train_X[i], self.train_y[i] = DG.generate_spirals(N=N, K=2, noise=1, density=0.3, rng=rng)
        # self.test_X[i], self.test_y[i] = DG.generate_spirals(N=N, K=2, noise=1, density=0.3, rng=rng)
        # self.Ctrain_X[i], self.Ctrain_y[i] = DG.generate_spirals(N=N, K=2, noise=1, density=0.3, rng=rng)
        # self.Ctest_X[i], self.Ctest_y[i] = DG.generate_spirals(N=N, K=2, noise=1, density=0.3, rng=rng)

    def load(self, fname='SimulationData.pickle', save=False):
        '''
        loads saved simulation dataset or saves current attributes as a pickle
        '''
        CLFPATH = os.path.join(os.getcwd(), 'clf\\')
        filename = CLFPATH + fname

        if os.path.exists(filename) and not save:
            with open(filename, 'rb') as f:
                dataset = pickle.load(f, encoding='bytes')

            for j in range(2):
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

# class datasetLoader:

#     def __init__(self):    
#         pass
    
