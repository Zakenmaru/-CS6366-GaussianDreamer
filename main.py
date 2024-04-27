import sys

from DiffusionModel3D import DiffusionModel3D
from GaussianInitialization import GaussianInitialization
from OptimizationWith2DModel import OptimizationWith2DModel

if __name__ == "__main__":

    prompt = sys.argv[1]
    sample_proportion = float(sys.argv[2])

    diffusion_model_3d = DiffusionModel3D(prompt)
    diffusion_model_3d.generateTriangleMesh()
    triangle_mesh = diffusion_model_3d.mesh

    print("Triangle mesh generated using 3d diffusion model")
    print("# of triangles in mesh: {0}".format(triangle_mesh.vertices.shape[0]))

    diffusion_model_3d.generatePointClouds(sample_proportion)
    point_clouds = diffusion_model_3d.point_clouds

    print("Point clouds generated with random weighted probabilities")
    print("# of point clouds: {0}".format(point_clouds.vertices.shape[0]))

    diffusion_model_3d.visualizePointClouds()
    diffusion_model_3d.visualizePointCloudsViews()

    gaussian_initialization = GaussianInitialization(point_clouds.vertices, point_clouds.colors)
    gaussians = gaussian_initialization.initializeGaussians()

    print("3D Gaussians initialized")

    diffusion_model_2d = OptimizationWith2DModel("a shark",
                                                 gaussians.positions,
                                                 gaussians.colors,
                                                 gaussians.scales,
                                                 gaussians.rotations,
                                                 gaussians.alphas)
    print("Optimizing with 2d Diffusion Model")
    diffusion_model_2d.train()

    print("Training iteration images saved to \"training_iterations.gif\"")
