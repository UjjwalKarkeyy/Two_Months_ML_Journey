import numpy as np
import scipy  
class Mahalanobis:
    # constructor
    def __init__(self, x = None, data = None, cov = None):
        self.x = x
        self.data = data
        self.cov = cov
    
    def cal_mahalanobis_dis(self):
        """
        x: (n_features,) or (n_samples, n_features)
        data: array-like or DataFrame of shape (n_samples, n_features)
        cov: optional covariance matrix (n_features, n_features)
        returns:
          - float if x is a single 1D vector
          - 1D array of distances if x is a 2D batch
        """
        x_minus_mu = self.x - np.mean(self.data)
        if self.cov is None:
            self.cov = np.cov(self.data.values.T)
            # self.data.values.T ensures np.cov computes covariance across features, not across samples.
        inv_covmat = scipy.linalg.inv(self.cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return mahal.diagonal()
        # computation produces a full matrix for multiple points, but we only need the self-distances (one per point).

    def greet(self):
        print('Hello from mahala file')