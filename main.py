import sys
from DiffusionModel3D import DiffusionModel3D

if __name__ == "__main__":

    prompt = sys.argv[1]

    diffusion_model_3d = DiffusionModel3D(prompt)
    diffusion_model_3d.generateTriangleMesh()
    triangle_mesh = diffusion_model_3d.mesh

    print(len(triangle_mesh.vertices))
    print(len(triangle_mesh.normals))
    print(len(triangle_mesh.colors))

    diffusion_model_3d.generatePointclouds()
    point_clouds = diffusion_model_3d.point_clouds

    print(len(point_clouds.vertices))
    print(len(point_clouds.normals))
    print(len(point_clouds.colors))