import numpy as np
from .dataset_generator import DatasetGenerator as DG

class train_test_load:

    def __init__(self):
        N = 750 #training sample size
        train_rng = 1 #training set range
        test_rng  = 3 #testing set range

        train_mean = np.array([-train_rng,-train_rng])

        self.gausX_train, self.gausY_train = DG.generate_gaussian_parity(N=N, mean=train_mean, cov_scale=0.1, angle_params=np.pi)
        self.CgausX_train, self.CgausY_train = DG.generate_gaussian_parity(N=N, mean=train_mean, cov_scale=0.1, angle_params=np.pi, cc=True)

        self.gausRX, self.gausRY = DG.generate_gaussian_parity(N=N, mean=train_mean, cov_scale=0.1, angle_params=np.pi/4)
        self.CgausRX, self.CgausRY = DG.generate_gaussian_parity(N=N, mean=train_mean, cov_scale=0.1, angle_params=np.pi/4, cc=True)

        self.gausSX, self.gausSY = DG.generate_gaussian_parity(N=N, mean=train_mean, cov_scale=0.01, angle_params=np.pi)
        self.CgausSX, self.CgausSY = DG.generate_gaussian_parity(N=N, mean=train_mean, cov_scale=0.01, angle_params=np.pi, cc=True)

        self.unifX, self.unifY = DG.generate_uniform_XOR(N=N, b=train_rng)
        self.CunifX, self.CunifY = DG.generate_uniform_XOR(N=N, b=train_rng, cc=True)

        self.spirX, self.spirY = DG.generate_spirals(N=N, K=2, noise=2.5, rng=train_rng)

class datasetLoader:

    def __init__(self):
        pass