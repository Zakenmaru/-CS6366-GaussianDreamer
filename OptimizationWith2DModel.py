import torch
import math

from gsplat import project_gaussians, rasterize_gaussians
from torch import Tensor, optim
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from torch.nn.functional import interpolate, mse_loss


# Implements the gradient descent optimization on the 2D diffusion model
class OptimizationWith2DModel:

    def __init__(self, prompt, positions, colors, scales, rotations, opacities):
        self.prompt = prompt
        self.initialize_parameters(positions, colors, scales, rotations, opacities)
        self.prepare_2d_diffusion_model()

    # Intialize all the required parameters for optimization
    def initialize_parameters(self, positions, colors, scales, rotations, opacities):
        self.positions = torch.tensor(positions, dtype=torch.float32, device='cuda')
        self.colors = torch.tensor(colors, dtype=torch.float32, device='cuda')
        self.scales = torch.tensor(scales, dtype=torch.float32, device='cuda')
        self.quats = torch.tensor(rotations, dtype=torch.float32, device='cuda')
        self.opacities = torch.tensor(opacities, dtype=torch.float32, device='cuda')

        # The parameters which will be tuned
        self.positions.requires_grad = True
        self.colors.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.opacities.requires_grad = True

        self.glob_scale = float(1.0)
        self.viewmat = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 5.0],
                                     [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda')
        self.fx = 0.5 * 512.0 / math.tan(0.5 * (math.pi / 2.0))
        self.fy = 0.5 * 512.0 / math.tan(0.5 * (math.pi / 2.0))
        self.cx = float(256.0)
        self.cy = float(256.0)
        self.img_height = 512
        self.img_width = 512
        self.block_width = 16

        self.scheduler = None
        self.pipe = None
        self.autoencoder = None
        self.denoiser = None

        self.prompt_embeddings = self.get_prompt_text_embeddings()

    # Prepare the 2d diffusion model and initialize the autoencoder (vae) and denoiser (unet)
    def prepare_2d_diffusion_model(self):
        self.scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base",
                                                                subfolder="scheduler",
                                                                torch_dtype=torch.float32)
        self.pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base",
                                                            tokenizer=None,
                                                            safety_checker=None,
                                                            feature_extractor=None,
                                                            requires_safety_checker=False,
                                                            torch_dtype=torch.float32).to("cuda")

        self.autoencoder = self.pipe.vae.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        self.denoiser = self.pipe.unet.eval()
        for param in self.denoiser.parameters():
            param.requires_grad_(False)

        self.weights = self.scheduler.alphas_cumprod.to("cuda")

    # Training pipeline to learn the positions, colors, scales, quats, and opacities
    def train(self):
        adam_optimizer = optim.Adam([{'params': [self.positions], 'lr': 0.00005},
                                     {'params': [self.colors], 'lr': 0.0125},
                                     {'params': [self.scales], 'lr': 0.001},
                                     {'params': [self.quats], 'lr': 0.01},
                                     {'params': [self.opacities], 'lr': 0.01}])

        self.training_images = []
        self.losses = []
        for iter in range(50):
            gaussians_pos_2d, depths, radii, conics, compensation, num_tiles_hit, cov_3d = project_gaussians(
                self.positions,
                self.scales,
                self.glob_scale,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.viewmat,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.img_height,
                self.img_width,
                self.block_width)

            rendered_image = rasterize_gaussians(gaussians_pos_2d,
                                                 depths,
                                                 radii,
                                                 conics,
                                                 num_tiles_hit,
                                                 self.colors,
                                                 self.opacities,
                                                 self.img_height,
                                                 self.img_width,
                                                 self.block_width)
            rendered_image = rendered_image.permute(2, 0, 1)
            rendered_image.unsqueeze_(0)

            latents = self.image_to_latents(rendered_image).to("cuda")

            t = torch.randint(20, 981, [1], dtype=torch.long, device="cuda")

            sds_gradient = self.calculate_sds_gradient(latents, t)
            sds_gradient = torch.nan_to_num(sds_gradient)

            sds_loss = mse_loss(latents, (latents - sds_gradient), reduction='sum')

            adam_optimizer.zero_grad()
            sds_loss.backward()
            adam_optimizer.step()

            transform = transforms.ToPILImage()
            img = transform(rendered_image.squeeze())
            self.training_images.append(img)

            print("Iteration: {0}/50   ---   Loss: {1}".format(iter, sds_loss.item()))

            self.losses.append(sds_loss)

        self.training_images[0].save("training_iterations.gif",
                                     save_all=True,
                                     append_images=self.training_images[1:],
                                     optimize=False,
                                     duration=10,
                                     loop=0)

    # Calculate the Score Distillation Sampling gradient by passing in the randomly noised image to the
    # 2d diffusion model and learning the denoised image (by predicting the noise itself)
    def calculate_sds_gradient(self, latents, t):
        with torch.no_grad():
            actual_noise = torch.randn(latents.size(), dtype=latents.dtype, layout=latents.layout, device="cuda")
            noisy_latents = self.scheduler.add_noise(latents, actual_noise, t)
            predicted_noise = self.denoiser_forward(noisy_latents,
                                                    t,
                                                    self.prompt_embeddings)

        sds_gradient = (1 - self.weights[t]).view(-1, 1, 1, 1) * (predicted_noise - actual_noise)

        return sds_gradient

    def image_to_latents(self, image):
        scaled_image = image * 2.0 - 1.0
        encoded_image = self.autoencoder.encode(scaled_image.to(torch.float32)).latent_dist
        latents = (encoded_image.sample() * 0.18215).to(torch.float32)
        return latents

    def get_prompt_text_embeddings(self):
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                  subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                     subfolder="text_encoder").to("cuda:0")
        for param in text_encoder.parameters():
            param.requires_grad_(False)

        tokens = tokenizer([self.prompt],
                           padding="max_length",
                           max_length=77,
                           return_tensors="pt")

        with torch.no_grad():
            text_embeddings = text_encoder(tokens.input_ids.to('cuda'))[0]

        del tokenizer
        del text_encoder

        return text_embeddings

    def denoiser_forward(self, latents, t, text_embeddings):
        return self.denoiser(latents.to(torch.float32),
                             t.to(torch.float32),
                             encoder_hidden_states=text_embeddings.to(torch.float32)).sample.to(torch.float32)
