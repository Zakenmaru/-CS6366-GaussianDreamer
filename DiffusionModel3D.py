import torch
from diffusers import ShapEPipeline
import numpy as np

from TriangleMesh import TriangleMesh

class DiffusionModel3D():

    def __init__(self, prompt):
        self.prompt = prompt
        self.mesh = None

    def generateTriangleMesh(self):
        pipe = ShapEPipeline.from_pretrained("openai/shap-e").to("cuda")

        guidance_scale = 15.0
        mesh_list = pipe(
            self.prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=64,
            frame_size=256,
            output_type="mesh"
        ).images
        self.convertMesh(mesh_list[0])

    def convertMesh(self, mesh):
        self.mesh = TriangleMesh()

        self.mesh.vertices = mesh.verts
        self.mesh.faces = mesh.faces
        self.mesh.colors = self.getColors(mesh.vertex_channels)
        self.getNormals()

    def getColors(self, color_dict):
        colors = []
        for i in range(len(color_dict['R'])):
            color_arr = np.array([color_dict['R'].cpu()[i], color_dict['G'].cpu()[i], color_dict['B'].cpu()[i]])
            colors.append(color_arr)

        return torch.tensor(colors)

    def getNormals(self):
        self.normals = np.zeros(self.mesh.vertices.shape)
        for v1_ind, v2_ind, v3_ind in self.mesh.faces:

            v1 = self.mesh.vertices[v1_ind].cpu()
            v2 = self.mesh.vertices[v2_ind].cpu()
            v3 = self.mesh.vertices[v3_ind].cpu()

            normal = np.cross(np.subtract(v2, v1), np.subtract(v3, v1))
            self.normals[v1_ind] = np.add(self.normals[v1_ind], normal)
            self.normals[v2_ind] = np.add(self.normals[v2_ind], normal)
            self.normals[v3_ind] = np.add(self.normals[v3_ind], normal)

        for i in range(len(self.normals)):
            norm = np.linalg.norm(normals[i])
            if norm != 0:
                self.normals[i] = np.divide(self.normals[i], norm)

        self.normals = torch.tensor(self.normals)