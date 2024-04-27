import math
import numpy as np
from scipy.spatial.distance import cdist

from Gaussians import Gaussians
from KDTree import KDTree
from BoundingBox import BoundingBox


# Initializes the 3D Gaussians from the point clouds after noisy point growing
# and color perturbation
class GaussianInitialization:
    def __init__(self, p_m, c_m):
        self.p_m = p_m
        self.c_m = c_m
        self.p_f = None
        self.c_f = None

    # Initialized Gaussians with the point clouds positions as centers
    # Colors of point clouds represented with spherical harmonics (degree=0)
    # Opacities of point clouds initialized to 0.1
    # Scale matrix obtained as log transformed avg distance of closest 3 points
    # Rotation matrix represented as quaternions with 1st value set to 1 and rest as 0
    def initializeGaussians(self):
        self.noisyPointGrowing()

        # opacity of 3D Gaussians
        alpha_b = np.full((len(self.p_f), 1), 0.75)

        # covariance matrix represented as scales and rotations

        # get 4 closest points (as their distances)
        # distances = cdist(self.p_f, self.p_f)
        # np.fill_diagonal(distances, np.inf)
        # closest_distances = np.partition(distances, 4)[:, :4]
        # mean_distances = np.mean(closest_distances[:, 1:] ** 2, axis=1)

        # scales = np.repeat(np.log(np.sqrt(mean_distances))[..., None], 3, axis=1)
        scales = np.full((len(self.p_f), 3), 0.75)

        rotations = np.zeros((len(self.p_f), 4))
        rotations[:, 0] = 1

        return Gaussians(self.p_f, self.c_f, alpha_b, scales, rotations)

    # Algorithm to perform noisy point growing and color perturbation
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
                # Color perturbation
                perturbed_color = np.array(self.c_m[i]) + 0.2 * np.random.random(size=3)

                # Represent color using spherical harmonics with degree=0
                # sh_coefficient = 0.5 * math.sqrt(1 / math.pi)
                # color = np.divide(np.subtract(pertrubed_color, 0.5), sh_coefficient)
                c_r.append(perturbed_color)

        p_r = np.array(p_r)
        c_r = np.array(c_r)

        self.p_f = np.concatenate((self.p_m, p_r))
        self.c_f = np.concatenate((self.c_m, c_r))