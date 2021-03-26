from datetime import datetime
from .dataset_generator import DatasetGenerator as DG


class IB:
    def __init__(self):
        self.date = datetime.now()
        self.spiral = DG.spiral_center(N=270, rng=1)

    def get_trueSpiral(self, l, r, h, sig=0.00008, rng=1, cc=True, spirals=120):
        '''
        get true posterior for spiral dataset
        '''
        return DG.true_spiral(l=l, r=r, h=h, sig=sig, rng=rng, cc=cc, spirals=spirals)

    def get_testpdfSpiral(self, N, K=2, noise=1, acorn=None, density=0.5, rng=1):
        '''
        get exploratory true posterior for spiral dataset (deprecated and replaced by get_trueSpiral)
        '''
        return DG.pdf_spiral(N, K, noise, acorn, density, rng)

    def get_trueXOR(self, l=-2, r=2, h=0.01, rotate=False, sig=0.25, cc=False):
        '''
        get true posterior for XOR dataset
        '''
        return DG.true_xor(l, r, h, rotate, sig, cc)

    def get_trueUXOR(self, l=-2, r=2, h=0.01, cc=False):
        '''
        get true posterior for uniform XOR dataset
        '''
        return DG.true_Uxor(l, r, h, cc)

    def get_colors(colors, inds):
        return [colors[i] for i in inds]

    def train_models():
        pass

