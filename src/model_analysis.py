'''
Subclass analysis module for inductive bias experiment

Author: Jong M. Shin
'''

import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import norm
from tqdm.notebook import tqdm
from .dataset_generator import DatasetGenerator as DG


class modelAnalysis(DG):

    def __init__(self):
        self.mask = DG.generate_mask()

    def _hellinger(self, p, q):
        '''
        Average point-wise hellinger distance between two discrete distributions
        p: a vector of posterior probability
        q: a vector of posterior probability
        '''
        try:
            # return np.sqrt(np.mean((np.sqrt(p) - np.sqrt(q)) ** 2))/ np.sqrt(2)
            # return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))/ np.sqrt(2)
            return np.mean(np.sqrt((np.sqrt(p) - np.sqrt(q)) ** 2) / np.sqrt(2))

        except Exception:
            print("Error: posterior size mismatch")

    # Validation of hellingner distance by implementation from gensim package

    # import gensim
    # from numpy.linalg import norm
    # from scipy.integrate import quad
    # from scipy import stats

    # gensim.matutils.hellinger(gaus_post[0][0:], gaus_post[1][0:])

    def _hellinger_explicit(self, p, q):
        '''
        Individual point-wise hellinger distance between two discrete distributions
        p: a vector of posterior probability
        q: a vector of posterior probability
        '''
        try:
            temp = []

            for p_i, q_i in zip(p, q):
                temp.append((np.sqrt(p_i) - np.sqrt(q_i)) ** 2)

            return np.sqrt(temp) / np.sqrt(2)

        except Exception:
            print("Error: posterior size mismatch")

    def _hellinger_cont(self, p, q):
        '''
        Hellinger distance for continuous probability distributions
        '''

        f = scipy.stats.gaussian_kde(p)
        g = scipy.stats.gaussian_kde(q)

        def integrand(x):
            return (f(x)**0.5 - g(x)**0.5)**2

        ans, err = quad(integrand, -np.inf, np.inf)
        return f, g, ans / 2

    def compute_hellinger(self, estP, trueP, fast=False):
        '''
        compute hellinger distance of p and q distribution
        '''
        hdist = []
        
        for k in tqdm(range(2), desc='hellinger', leave=False):
            hdist.append([])
            for i in tqdm(range(5), desc='helliner-inner', leave=False):
                hdist[k].append([])
                for j in range(4):
                    hdist[k][i].append([])
                    if k == 0: continue
                    if i == 2 or i == 4:
                        hdist[k][i][j] = self._hellinger_explicit(
                            trueP[k][i][2], estP[k][i][j])

        return hdist

    def _euclidean(self, dat, origin=[0, 0]):
        '''
        compute euclidean distance from the origin

        dat: a N x M matrix where the first two columns must have x,y coordinates of the grid
        origin: origin of the euclidean distance

        output
        ------
        returns a vector of euclidean distance from the origin
        '''
        dat = np.array(dat)

        new_dat = np.sqrt((dat[:, 0] - origin[0])**2
                          + (dat[:, 1] - origin[1])**2)
        return new_dat

    def smooth_radial_distance(self, dat, origin=[0, 0], step=0.1, end=3, srng=0.5, verbose=False, **kwargs):
        '''
        compute posterior on grid

        dat: a N x M x 3 matrix where there are N number of ML models,
        M number of rows of data, and three columns of x, y, posterior
        origin: origin of the distance
        step: step in which the search radius is increasing
        end: outer boundary of the search
        srng: smoothing range (aka width of the search ring)
        '''
        h_rad = np.arange(0, end+step, step)
        alpha = step * srng  # line vicinity (0.5), aka smoothing range
        new_rad = []
        new_dat = []

        n = dat.shape[0]
        distC = self._euclidean(dat, origin=origin)  # euclidean distance
        idx = np.arange(0, n)
        cin, cout = 0, 0

        for r in h_rad:  # radius

            temp_distC = distC <= r

            tempidx = idx[temp_distC]
            tempidx = idx[np.all((distC <= r+alpha, distC >= r-alpha), axis=0)]

            # distC[temp_distC] = 10.  # prevent double counting

            if tempidx.size != 0:
                new_rad.append(r)
                new_dat.append(dat[:, 2][tempidx].astype(float))

                if r > 1:
                    cout += tempidx.size
                else:
                    cin += tempidx.size

        if verbose:
            print('\n' + '#'*20
                  + '\n# outside: ' + str(cout)
                  + '\n# inside:  ' + str(cin) + '\n' + '#'*20)
        
        return [new_rad, new_dat]

    def _gauss2d(self, x, y, x0, y0, A, sig_x=1, sig_y=1):
        '''
        2D Gaussian function
        '''
        frac_x = (x-x0)**2 / (2*sig_x**2)
        frac_y = (y-y0)**2 / (2*sig_y**2)
        out = A*np.exp(-(frac_x+frac_y))
        return out

    @staticmethod
    def sample(dat, target, N):
        '''
        Randomly samples from the target and append on dat, thus the dat and the target must be of the same dimension
        '''
        temp_idx = np.arange(0, target.shape[0],1)
        idx_selected = np.random.choice(temp_idx, N)

        return np.vstack([dat, target[idx_selected]])

    def smooth_gaussian_distance(self, dat, radius=1, step=0.5, method='mean'):
        '''
        Applies gaussian smoothing over a grid. Takes N x 3 matrix where first two columns are X and Y coordinates of the grid and the last column is variable of interest.

        dat: N x 3 matrix
        radius: radius of circular ROI where gaussian smoothing will be applied (default: 1)
        step: sparsity of circular ROI center (default: 0.5)
        method: method of smoothing, options='mean', 'var' (default: mean)

        output
        ------
        List of (X,Y) coordinates of circular ROI center and smoothed gaussian variable of interest
        '''
        grid = dat[:,:2]
        xL, xR = min(dat[:,0]), max(dat[:,0])
        yT, yB = min(dat[:,1]), max(dat[:,1])

        X = np.arange(xL, xR, step).round(1)
        Y = np.arange(yT, yB, step).round(1)

        XY = list(product(X,Y))

        out = []

        for i in tqdm(range(len(XY)),leave=False):
            x = dat[self._euclidean(dat, XY[i]) < radius][:,0]
            y = dat[self._euclidean(dat, XY[i]) < radius][:,1]
            a = dat[self._euclidean(dat, XY[i]) < radius][:,2]
            if method == 'mean':
                out.append(self._gauss2d(x=x, y=y, x0=XY[i][0], y0=XY[i][1], A=a).mean())
            elif method == 'var':
                out.append(self._gauss2d(x=x, y=y, x0=XY[i][0], y0=XY[i][1], A=a).var())

        return [XY, out]

    @staticmethod
    def pointwise_gridAverage(dat):
        '''
        Averages the values associated with the grid point-wise.

        Takes N x 3 array where first two columns are assumed to be the coordinates. 
        They are joined into one column and any duplicates are removed based on this column.

        output
        ------
        N x 3 array where the structure of the input is maintained and duplicates are removed
        '''
        dat = pd.DataFrame(dat, columns=['x', 'y', 'c']).astype(float)
        dat['xy'] = dat['x'].astype(str).str.cat(dat['y'].astype(str),sep=',')
        dat = dat.groupby('xy').mean().reset_index(drop=True)
        
        return dat