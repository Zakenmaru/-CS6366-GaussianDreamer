import sys
from DiffusionModel3D import DiffusionModel3D

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