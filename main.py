import sys
from DiffusionModel3D import DiffusionModel3D

if __name__ == "__main__":

    prompt = sys.argv[1]

    diffusion_model_3d = DiffusionModel3D(prompt)
    triangle_mesh = diffusion_model_3d.generateTriangleMesh()

    print(triangle_mesh)