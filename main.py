import sys
import numpy as np

from DiffusionModel3D import DiffusionModel3D
from GaussianInitialization import GaussianInitialization
from Gaussians import Gaussians

if __name__ == "__main__":

    prompt = sys.argv[1]

    diffusion_model_3d = DiffusionModel3D(prompt)
    diffusion_model_3d.generateTriangleMesh()
    triangle_mesh = diffusion_model_3d.mesh

    print(triangle_mesh.vertices.shape)
    print(triangle_mesh.normals.shape)
    print(triangle_mesh.colors.shape)

    diffusion_model_3d.generatePointclouds()
    point_clouds = diffusion_model_3d.point_clouds

    print(point_clouds.vertices.shape)
    print(point_clouds.normals.shape)
    print(point_clouds.colors.shape)

    gaussian_initialization = GaussianInitialization(point_clouds.vertices, point_clouds.colors)
    gaussians = gaussian_initialization.initializeGaussians()
    print(gaussians.positions.shape)
    print(gaussians.colors.shape)
    print(gaussians.covariance.shape)
    print(gaussians.alpha.shape)