'''
Subclass analysis module for inductive bias experiment

Author: Jong M. Shin
'''

import numpy as np
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

    def compute_hellinger(self, estP, trueP):
        '''
        compute hellinger distance of p and q distribution
        '''
        hdist = []

        for k in tqdm(range(2)):
            hdist.append([])
            for i in tqdm(range(5), leave=False):
                hdist[k].append([])
                for j in range(4):
                    hdist[k][i].append([])
                    hdist[k][i][j] = self._hellinger_explicit(
                        trueP[k][i][2], estP[k][i][j][:, 0])

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
        h_rad = np.arange(0, end, step)
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

            distC[temp_distC] = 10.  # prevent double counting

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

    # def compute_hellinger(self, estP, trueP):

    #     '''
    #     compute hellinger distance of p and q distribution
    #     '''

    #     # if path.exists(filename) and save:
    #     #     with open(filename, 'rb') as f:
    #     #         WHOLE_hellinger, OUT_hellinger, IN_hellinger, uX_inside, uX_outside = pickle.load(f, encoding='bytes')
    #     #     print('[', filename, '] loaded')

    #     # else:
    #     uX_outside = []
    #     uX_inside  = []
    #     OUT_hellinger = []
    #     IN_hellinger  = []
    #     WHOLE_hellinger = []
    #     uX = self.mask

    #     # split inside and outside
    #     for i in range(len(uX)):

    #         if np.sqrt((uX[i]**2).sum(axis=0)) < 1:  #np.all([uX[i] > -1, uX[i] < 1]):
    #             uX_inside.append(uX[i].tolist())
    #         else:
    #             uX_outside.append(uX[i].tolist())

    #     uX_outside = np.array(uX_outside)
    #     uX_inside  = np.array(uX_inside)

    #     for dat in tqdm(range(len(estP))): #number of datasets (=ib.dtype)

    #         # clear_output(wait=True)

    #         OUT_hellinger.append([])
    #         IN_hellinger.append([])
    #         WHOLE_hellinger.append([])

    #         # calculate RADIAL hellinger distance
    #         for i in range(len(estP[0])): #number of ML models (=ib.mtype)

    #             inner_trueP = np.zeros(len(uX_inside))
    #             inner_testP = np.zeros(len(uX_inside))
    #             outer_trueP = np.zeros(len(uX_outside))
    #             outer_testP = np.zeros(len(uX_outside))
    #             innermask = []
    #             outermask = []
    #             temp_OUT_hellinger = np.zeros(len(uX_outside))
    #             temp_IN_hellinger  = np.zeros(len(uX_inside))
    #             c = 0
    #             cc = 0
    #             WHOLE_hellinger[dat].append([])

    #             for ii, prob in enumerate(uX):
    #                 if np.sqrt((prob**2).sum(axis=0)) < 1: #np.all([prob > -1, prob < 1]):
    #                     inner_trueP[c] = trueP[dat][ii]
    #                     inner_testP[c] = estP[dat][i][ii]
    #                     nantest = self.hellinger(inner_trueP[c], inner_testP[c])
    #                     if str(nantest) != 'nan':
    #                         temp_IN_hellinger[c] = nantest
    #                     else:
    #                         print('There is NaN in {}th data at location {}'.format(i, ii))
    #                         break
    #                     WHOLE_hellinger[dat][i].append(temp_IN_hellinger[c])
    #                     innermask.append(prob)
    #                     c += 1
    #                 else:
    #                     print(inner_trueP[c] , trueP[dat][ii])
    #                     outer_trueP[cc] = trueP[dat][ii]
    #                     outer_testP[cc] = estP[dat][i][ii]
    #                     temp_OUT_hellinger[cc] = self.hellinger(outer_trueP[cc], outer_testP[cc])
    #                     WHOLE_hellinger[dat][i].append(temp_OUT_hellinger[cc])
    #                     outermask.append(prob)
    #                     cc += 1

    #             OUT_hellinger[dat].append(temp_OUT_hellinger)
    #             IN_hellinger[dat].append(temp_IN_hellinger)

    #     WHOLE_hellinger = np.array(WHOLE_hellinger)

    #     # with open(filename, 'wb') as f:
    #     #     pickle.dump([WHOLE_hellinger, OUT_hellinger, IN_hellinger, uX_inside, uX_outside], f)

    #     print ('\n' + '#'*40
    #     + '\nTotal number of points on grid: ' + str(len(OUT_hellinger[0][0])+len(IN_hellinger[0][0]))
    #     + '\nOuter number of points on grid: ' + str(len(OUT_hellinger[0][0]))
    #     + '\nInner number of points on grid: ' + str(len(IN_hellinger[0][0])))

    #     return [WHOLE_hellinger, OUT_hellinger, IN_hellinger, uX_inside, uX_outside]
