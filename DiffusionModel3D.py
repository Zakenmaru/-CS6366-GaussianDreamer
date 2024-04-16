import torch
from diffusers import ShapEPipeline
import numpy as np

from TriangleMesh import TriangleMesh
from PointClouds import PointClouds

class DiffusionModel3D():

    def __init__(self, prompt):
        self.prompt = prompt
        self.mesh = TriangleMesh()
        self.vertices = None
        self.faces = None
        self.colors = None
        self.point_clouds = PointClouds()

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
        self.vertices = mesh.verts.cpu()
        self.faces = mesh.faces
        self.colors = self.getColors(mesh.vertex_channels)
        self.getNormals()

        self.getTriangleMesh()

    def getTriangleMesh(self):
        for v1_ind, v2_ind, v3_ind in self.faces:
            self.mesh.vertices.append([self.vertices[v1_ind], self.vertices[v2_ind], self.vertices[v3_ind]])
            self.mesh.colors.append([self.colors[v1_ind], self.colors[v2_ind], self.colors[v3_ind]])
            self.mesh.normals.append([self.normals[v1_ind], self.normals[v2_ind], self.normals[v3_ind]])

        self.mesh.vertices = np.array(self.mesh.vertices)
        self.mesh.colors = np.array(self.mesh.colors)
        self.mesh.normals = np.array(self.mesh.normals)

    def getColors(self, color_dict):
        colors = []
        for i in range(len(color_dict['R'])):
            color_arr = np.array([color_dict['R'].cpu()[i], color_dict['G'].cpu()[i], color_dict['B'].cpu()[i]])
            colors.append(color_arr)

        return torch.tensor(colors)

    def getNormals(self):
        self.normals = np.zeros(self.vertices.shape)
        for v1_ind, v2_ind, v3_ind in self.faces:

            v1 = self.vertices[v1_ind].cpu()
            v2 = self.vertices[v2_ind].cpu()
            v3 = self.vertices[v3_ind].cpu()

            normal = np.cross(np.subtract(v2, v1), np.subtract(v3, v1))
            self.normals[v1_ind] = np.add(self.normals[v1_ind], normal)
            self.normals[v2_ind] = np.add(self.normals[v2_ind], normal)
            self.normals[v3_ind] = np.add(self.normals[v3_ind], normal)

        for i in range(len(self.normals)):
            norm = np.linalg.norm(self.normals[i])
            if norm != 0:
                self.normals[i] = np.divide(self.normals[i], norm)

        self.normals = torch.tensor(self.normals)

    def generatePointclouds(self):
        triangle_areas = self.calcTriangleAreas(self.mesh.vertices[:,0,:],
                                                self.mesh.vertices[:,1,:],
                                                self.mesh.vertices[:,2,:])

        randomly_weighted_vertices = np.random.choice(range(len(triangle_areas)),
                                                      size=15000,
                                                      p=triangle_areas / sum(triangle_areas))

        self.mesh.vertices = self.mesh.vertices[randomly_weighted_vertices]
        self.mesh.normals = self.mesh.normals[randomly_weighted_vertices]
        self.mesh.colors = self.mesh.colors[randomly_weighted_vertices]

        u = np.random.rand(15000, 1)
        v = np.random.rand(15000, 1)
        u[u+v>1] = 1 - u[u+v>1]
        v[u+v>1] = 1 - v[u+v>1]

        self.point_clouds.vertices = ((self.mesh.vertices[:,0,:] * u)
                                      + (self.mesh.vertices[:,1,:] * v)
                                      + ((1 - u - v) * self.mesh.vertices[:,2,:])).astype(np.float32)
        self.point_clouds.colors = ((self.mesh.colors[:, 0, :] * u)
                                      + (self.mesh.colors[:, 1, :] * v)
                                      + ((1 - u - v) * self.mesh.colors[:, 2, :])).astype(np.float32)

        mesh_normal_sum = self.mesh.normals[:,0,:] + self.mesh.normals[:,1,:] + self.mesh.normals[:,2,:]
        self.point_clouds.normals = mesh_normal_sum / np.linalg.norm(mesh_normal_sum, axis=1)[...,None].astype(np.float32)


    def calcTriangleAreas(self, v1_arr, v2_arr, v3_arr):
        return np.linalg.norm(np.cross(v2_arr - v1_arr, v3_arr - v1_arr), axis=1)
