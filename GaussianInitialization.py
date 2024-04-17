import numpy as np
from scipy.spatial.distance import cdist

from KDTree import KDTree
from BoundingBox import BoundingBox
from Gaussians import Gaussians

class GaussianInitialization:
    def __init__(self, p_m, c_m):
        self.p_m = p_m
        self.c_m = c_m
        self.p_f = None
        self.c_f = None

    def initializeGaussians(self):
        self.noisyPointGrowing()

        # get distance between nearest two positions
        distances = cdist(self.p_f, self.c_f)
        np.fill_diagonal(distances, np.inf)

        # distance between nearest two positions
        D = np.min(distances[np.nonzero(distances)])

        # covariance and opacity of 3D Gaussians
        sigma_b = np.full((len(self.p_f), 1), D)
        alpha_b = np.full((len(self.p_f), 1), 0.1)

        return Gaussians(self.p_f, self.c_f, sigma_b, alpha_b)

    def noisyPointGrowing(self):
        K_m = KDTree(self.p_m, dim=3)

        BBox = BoundingBox(self.p_m)
        low, high = BBox.min_bound, BBox.max_bound

        ps_u = np.random.uniform(low, high, size=(len(self.p_m), BBox.dim))

        p_r, c_r = [], []
        for p_u in ps_u:
            p_un, i = K_m.nearest_neighbor(p_u)

            dist = np.linalg.norm(p_un - p_u)
            max_dist = np.linalg.norm(p_un) + np.linalg.norm(p_u)
            norm_dist = dist / max_dist

            if abs(norm_dist) < 0.01:
                p_r.append(p_u)
                c_r.append(list(np.array(self.c_m[i]) + 0.2 * np.random.random(size=3)))

        p_r = np.array(p_r)
        c_r = np.array(c_r)

        self.p_f = np.concatenate((self.p_m, p_r))
        self.c_f = np.concatenate((self.c_m, c_r))
