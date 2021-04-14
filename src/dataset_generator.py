'''
Static class dataset generator for inductive bias experiment

Author: Jong M. Shin
'''

import numpy as np
from scipy.stats import norm
from tqdm.notebook import tqdm_notebook as tqdm  # compatible with jupyter


class DatasetGenerator:

    @staticmethod
    def generate_2d_rotation(theta=0, acorn=None):
        if acorn is not None:
            np.random.seed(acorn)

        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        return R

    @staticmethod
    def generate_gaussian_parity(n, mean=np.array([-1, -1]), cov_scale=1, angle_params=None, k=1, acorn=None, contain=True, cc=False, train=True):
        if acorn is not None:
            np.random.seed(acorn)

        d = len(mean)
        lim = abs(mean[0])
        if train:
            unitBound = 1
        else:
            unitBound = lim

        mean = mean + 0.5 * abs(mean[0])  # adjusting gaussian center

        mnt = np.random.multinomial(n, 1/(4**k) * np.ones(4**k))
        cumsum = np.cumsum(mnt)
        cumsum = np.concatenate(([0], cumsum))

        Y = np.zeros(n)
        X = np.zeros((n, d))

        for i in range(2**k):
            for j in range(2**k):
                temp = np.random.multivariate_normal(mean, cov_scale * np.eye(d),
                                                     size=mnt[i*(2**k) + j])
                temp[:, 0] += i*lim
                temp[:, 1] += j*lim

                # screen out values outside the boundary
                if contain:
                    if cc:
                        # circular bbox
                        idx_oob = np.where(
                            np.sqrt((temp**2).sum(axis=1)) > unitBound)

                        for l in idx_oob:

                            while True:
                                temp2 = np.random.multivariate_normal(
                                    mean, cov_scale * np.eye(d), size=1)

                                if np.sqrt((temp2**2).sum(axis=1)) < unitBound:
                                    temp[l] = temp2
                                    break
                    else:
                        # square bbox
                        idx_oob = np.where(abs(temp) > unitBound)

                        for l in idx_oob:

                            while True:
                                temp2 = np.random.multivariate_normal(
                                    mean, cov_scale * np.eye(d), size=1)

                                if (abs(temp2) < unitBound).all():
                                    temp[l] = temp2
                                    break

                X[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = temp

                if i % 2 == j % 2:
                    Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 0
                else:
                    Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 1

        if d == 2:
            if angle_params is None:
                angle_params = np.random.uniform(0, 2*np.pi)

            R = DatasetGenerator.generate_2d_rotation(angle_params)
            X = X @ R

        else:
            raise ValueError('d=%i not implemented!' % (d))

        return X, Y.astype(int)

    @staticmethod
    def generate_spirals(N=100, K=5, noise=0.5, acorn=None, density=0.3, rng=1, cc=None):
        '''
        method that generates simulation spiral dataset
        '''
        # N number of points per class
        # K number of classes
        X = []
        Y = []

        size = int(N/K)  # equal number of points per feature

        if acorn is not None:
            np.random.seed(acorn)

        if K == 2:
            turns = 2
        elif K == 3:
            turns = 2.5
        elif K == 5:
            turns = 3.5
        elif K == 7:
            turns = 4.5
        elif K == 1:
            turns = 1
        else:
            print("sorry, can't currently surpport %s classes " % K)
            return

        mvt = np.random.multinomial(N, 1/K * np.ones(K))

        if K == 2:
            r = np.random.uniform(0, rng, size=size)
            r = np.sort(r)
            t = np.linspace(0,  np.pi * 4 * rng, size) + \
                noise * np.random.normal(0, density, size)
            dx = r * np.cos(t)
            dy = r * np.sin(t)

            X.append(np.vstack([dx, dy]).T)
            X.append(np.vstack([-dx, -dy]).T)
            Y += [0] * size
            Y += [1] * size
        else:
            for j in range(1, K+1):
                r = np.linspace(0.01, rng, int(mvt[j-1]))
                t = np.linspace((j-1) * np.pi * 4 * turns/K,  j * np.pi * 4 * turns/K,
                                int(mvt[j-1])) + noise * np.random.normal(0, density, int(mvt[j-1]))
                dx = r * np.cos(t)
                dy = r * np.sin(t)

                dd = np.vstack([dx, dy]).T
                X.append(dd)
                # label
                Y += [j-1] * int(mvt[j-1])
        return np.vstack(X), np.array(Y).astype(int)

    @staticmethod
    def generate_mask(rng=3, h=0.01):
        '''
        method that generates the grid in the range of [-rng, rng] at h step
        '''
        l, r = -rng, rng
        x = np.arange(l, r, h)
        y = np.arange(l, r, h)
        x, y = np.meshgrid(x, y)
        sample = np.c_[x.ravel(), y.ravel()]

        return sample  # [:,0], sample[:,1]

    @staticmethod
    def xor_pdf(x, rotate=False, sig=0.25):
        '''
        method that draws gaussian posterior for xor true posterior
        '''
        # Generates true XOR posterior
        if rotate:
            mu01 = np.array([-0.5, 0])
            mu02 = np.array([0.5, 0])
            mu11 = np.array([0, 0.5])
            mu12 = np.array([0, -0.5])
        else:
            mu01 = np.array([-0.5, 0.5])
            mu02 = np.array([0.5, -0.5])
            mu11 = np.array([0.5, 0.5])
            mu12 = np.array([-0.5, -0.5])
        cov = sig * np.eye(2)
        inv_cov = np.linalg.inv(cov)

        p0 = (
            np.exp(-(x - mu01)@inv_cov@(x-mu01).T)
            + np.exp(-(x - mu02)@inv_cov@(x-mu02).T)
        )/(2*np.pi*np.sqrt(np.linalg.det(cov)))

        p1 = (
            np.exp(-(x - mu11)@inv_cov@(x-mu11).T)
            + np.exp(-(x - mu12)@inv_cov@(x-mu12).T)
        )/(2*np.pi*np.sqrt(np.linalg.det(cov)))

        # return p0-p1
        return p1/(p0+p1)

    @staticmethod
    def spiral_pdf(x, sig=0.25, rng=1, spirals=270):
        '''
        method that draws gaussian posterior at each spiral center
        '''
        x0, x1 = DatasetGenerator.spiral_center(spirals, K=2, rng=rng)

        mu01 = x0
        mu02 = x1
        cov = sig * np.eye(2)
        inv_cov = np.linalg.inv(cov)

        p0, p1 = 0, 0
        for mu in mu01:
            p0 += np.exp(-(x - mu)@inv_cov@(x-mu).T)
        for mu in mu02:
            p1 += np.exp(-(x - mu)@inv_cov@(x-mu).T)

        p0 = p0/(2*np.pi*np.sqrt(np.linalg.det(cov)))
        p1 = p1/(2*np.pi*np.sqrt(np.linalg.det(cov)))

        return p0/(p0+p1)

    @staticmethod
    def true_spiral(h=0.01, sig=0.00008, rng=4.3, cc=False, spirals=270, **kwarg):
        '''
        method that generates true posterior for spiral dataset
        '''
        X = DatasetGenerator.generate_mask(rng=rng, h=h)
        z = np.zeros(len(X), dtype=float)
        z[:] = 0.5

        for ii, x in enumerate(tqdm(X, leave=False)):
            if np.any([x <= -1.0, x >= 1.0]) and cc == False:  # or x.any() > 1
                pass
            elif np.sqrt((x**2).sum(axis=0)) > 1 and cc == True:
                pass
            else:
                nantest = DatasetGenerator.spiral_pdf(
                    x, sig=sig, rng=3, spirals=spirals)
                if str(nantest) != 'nan':
                    z[ii] = 1-nantest
                else:
                    # print ('There is {} at {} {}'.format(nantest, ii, x))
                    pass

        z = (z - min(z)) / (max(z) - min(z))

        return X[:, 0], X[:, 1], z

    @staticmethod
    def spiral_center(N, K=2, rng=3, **kwargs):
        '''
        method that generates spiral centers where gaussian is drawn at each center
        '''
        # N number of points per class
        # K number of classes
        X0, X1 = [], []

        turns = 2

        size = int(N/K)  # equal number of points per feature

        r = np.linspace(0, rng, size)
        # r = np.sort(r)
        t = np.linspace(0, np.pi * 4 * rng, size)
        dx = r * np.cos(t)
        dy = r * np.sin(t)

        X0.append(np.vstack([dx, dy]).T)
        X1.append(np.vstack([-dx, -dy]).T)

        return np.vstack(X0), np.vstack(X1)

    @staticmethod
    def true_xor(rng=3, h=0.01, rotate=False, sig=0.25, cc=False, **kwarg):
        '''
        method that generates true posterior for xor datasets
        '''
        X = DatasetGenerator.generate_mask(rng=rng, h=h)

        z = np.zeros(len(X), dtype=float)

        for ii, x in enumerate(X):
            if np.any([x <= -1.0, x >= 1.0]) and cc == False:  # or x.any() > 1
                z[ii] = 0.5
            elif np.sqrt((x**2).sum(axis=0)) > 1 and cc == True:
                z[ii] = 0.5
            else:
                z[ii] = 1-DatasetGenerator.xor_pdf(x, rotate=rotate, sig=sig)
            # z[ii] = 1-xor_pdf(x, rotate=rotate, sig=sig)

        z = (z - min(z)) / (max(z) - min(z))

        return X[:, 0], X[:, 1], z

    @staticmethod
    def true_Uxor(rng=3, h=0.01, cc=False, **kwarg):
        '''
        method that generates true posterior for uniform xor dataset
        '''
        X = DatasetGenerator.generate_mask(rng=rng, h=h)

        l = -1
        r = 1

        z = np.zeros(len(X), dtype=float) + 0.5

        for i, loc in enumerate(X):
            X0 = loc[0]
            X1 = loc[1]

            if X0 > l and X0 < 0 and X1 < r and X1 > 0:
                z[i] = 1
            elif X0 > 0 and X0 < r and X1 < r and X1 > 0:
                z[i] = 0
            elif X0 > l and X0 < 0 and X1 < 0 and X1 > l:
                z[i] = 0
            elif X0 > 0 and X0 < r and X1 < 0 and X1 > l:
                z[i] = 1

            if np.sqrt((np.c_[X0, X1]**2).sum(axis=1)) > 1 and cc == True:
                z[i] = 0.5

        return X[:, 0], X[:, 1], z

    @staticmethod
    def generate_uniform_XOR(b=1, N=750, cc=False, train=True):
        '''
        method that generates simulation uniform xor dataset
        '''
        boundary = np.random.multinomial(N, [1/4.]*4)
        bcum = np.cumsum(boundary)

        X = np.array([[0, 0]])
        Y = np.zeros(N)
        Y[bcum[0]:bcum[2]] = 1
        ol = 0.0  # degree of overlap

        if train:
            unitBound = 1
        else:
            unitBound = b

        for i in range(2):
            for j in range(2):

                idx = 2*i+j

                if i == 1:
                    tempX = np.random.uniform(ol, -b, boundary[idx])
                else:
                    tempX = np.random.uniform(-ol, b, boundary[idx])

                if j == 1:
                    tempY = np.random.uniform(ol, -b, boundary[idx])
                else:
                    tempY = np.random.uniform(-ol, b, boundary[idx])

                temp = np.c_[tempX, tempY]

                if cc:
                    # circular bbox
                    idx_oob = np.where(
                        np.sqrt((temp**2).sum(axis=1)) > unitBound)

                    for l in idx_oob:

                        while True:

                            if i == 1:
                                tempX = np.random.uniform(ol, -b, 1)
                            else:
                                tempX = np.random.uniform(-ol, b, 1)

                            if j == 1:
                                tempY = np.random.uniform(ol, -b, 1)
                            else:
                                tempY = np.random.uniform(-ol, b, 1)

                            temp2 = np.c_[tempX, tempY]

                            if np.sqrt((temp2**2).sum(axis=1)) < unitBound:
                                temp[l] = temp2
                                break

                    X = np.concatenate((X, temp))

                else:
                    X = np.concatenate((X, np.c_[tempX, tempY]))

        return X[1:], Y.astype('int')
