import torch
from diffusers import ShapEPipeline

class DiffusionModel3D():

    def __init__(self, prompt):
        self.prompt = prompt
        self.generateTriangleMesg(self.prompt)

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
        mesh = mesh_list[0]

        return mesh

