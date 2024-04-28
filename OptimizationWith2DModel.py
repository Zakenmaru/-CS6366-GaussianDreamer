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
        self.get_ground_truth_image()

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
                                     [0.0, 0.0, 0.0, 1.0]],
                                    dtype=torch.float32,
                                    device="cuda")
        self.viewmat.requires_grad = False
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

        # Get the ground truth image by projecting the 3d point cloud onto a 2d viewport
    def get_ground_truth_image(self):
        with torch.no_grad():
            # Perspective projection
            eye_fov = 60.0 * math.pi / 180.0
            zNear = -0.1
            zFar = -50

            t_val = abs(zNear) * math.tan(eye_fov / 2)
            r_val = t_val
            l_val = -r_val
            b_val = -t_val
            perspmat = torch.tensor([[2 * zNear / (r_val - l_val), 0.0, -(r_val + l_val) / (r_val - l_val), 0.0],
                                     [0.0, 2 * zNear / (t_val - b_val), -(t_val + b_val) / (t_val - b_val), 0.0],
                                     [0.0, 0.0, (zNear + zFar) / (zNear - zFar),
                                      -2 * zNear * zFar / (zNear - zFar)],
                                     [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device='cuda')

            # Viewport transformation
            viewportmat = torch.tensor([[256.0, 0.0, 0.0, 256.0],
                                        [0.0, 256.0, 0.0, 256.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda')

            # Get perspective projection matrix after applying view transformation
            viewmat = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0]],
                                   dtype=torch.float32,
                                   device="cuda")
            perspective_projection_matrix = torch.matmul(perspmat, viewmat).to("cuda")

            ones = torch.ones((self.positions.shape[0], 1)).to("cuda")
            positions = torch.cat((self.positions, ones), 1).to("cuda")
            apply_mat = lambda positions: torch.matmul(perspective_projection_matrix, positions)
            perspective_positions = torch.stack([apply_mat(x) for x in positions])[:, :3].to("cuda")

            # normalize the positions between -1 and 1
            perspective_positions -= perspective_positions.min()
            perspective_positions /= perspective_positions.max()
            perspective_positions = 2 * perspective_positions - 1
            perspective_positions = torch.cat((perspective_positions, ones), 1)

            # Apply viewport tranformation
            apply_viewportmat = lambda perspective_positions: torch.matmul(viewportmat, perspective_positions)
            ground_truth_positions = torch.stack([apply_viewportmat(x) for x in perspective_positions])[:, :3].to(
                "cuda")

            # Z-buffer algorithm to find colors of each pixel
            self.ground_truth_image = torch.ones((3, 512, 512)).to("cuda")
            z_buffer = torch.full((512, 512), math.inf)
            for i in range(len(self.positions)):
                x, y, z = ground_truth_positions[i][0], ground_truth_positions[i][1], ground_truth_positions[i][2]
                x = math.floor(x) - 1
                y = (511 - math.floor(y))

                if (z <= z_buffer[x][y]):
                    self.ground_truth_image[0][x][y] = self.colors[i][0]
                    self.ground_truth_image[1][x][y] = self.colors[i][1]
                    self.ground_truth_image[2][x][y] = self.colors[i][2]
                    z_buffer[x][y] = z

            transform = transforms.ToPILImage()
            img = transform(self.ground_truth_image)
            img.save("GroundTruthImage.png")

                # Prepare the 2d diffusion model and initialize the autoencoder (vae) and denoiser (unet)

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

    # Training pipeline to learn the positions, colors, scales, quats, and opacities from the ground
    # truth image
    def train_ground_truth(self):
        adam_optimizer = optim.Adam([{'params': [self.positions], 'lr': 0.00005},
                                     {'params': [self.colors], 'lr': 0.0125},
                                     {'params': [self.scales], 'lr': 0.001},
                                     {'params': [self.quats], 'lr': 0.01},
                                     {'params': [self.opacities], 'lr': 0.01}])

        self.ground_truth_training_images = []
        for iter in range(6000):
            adam_optimizer.zero_grad()
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

            loss = mse_loss(rendered_image, self.ground_truth_image)
            loss.backward()
            adam_optimizer.step()

            if (iter % 100 == 0):
                transform = transforms.ToPILImage()
                img = transform(rendered_image)
                self.ground_truth_training_images.append(img)

                print("Iteration: {0}/6000   ---   Loss: {1}".format(iter, loss))


        self.ground_truth_training_images[0].save("ground_truth_training_iterations.gif",
                                                  save_all=True,
                                                  append_images=self.ground_truth_training_images[1:],
                                                  optimize=False,
                                                  duration=10,
                                                  loop=0)

    # Training pipeline to learn the positions, colors, scales, quats, and opacities
    # using the 2d diffusion model
    def train(self):
        adam_optimizer = optim.Adam([{'params': [self.positions], 'lr': 0.00005},
                                     {'params': [self.colors], 'lr': 0.0125},
                                     {'params': [self.scales], 'lr': 0.001},
                                     {'params': [self.quats], 'lr': 0.01},
                                     {'params': [self.opacities], 'lr': 0.01}])

        self.sds_training_images = []
        for iter in range(1000):
            adam_optimizer.zero_grad()
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
            sds_loss.backward()
            adam_optimizer.step()
            if (iter % 50 == 0):
                transform = transforms.ToPILImage()
                img = transform(rendered_image.squeeze())
                self.sds_training_images.append(img)

                print("Iteration: {0}/1000   ---   Loss: {1}".format(iter, sds_loss))

        self.sds_training_images[0].save("diffusion_training_iterations.gif",
                                         save_all=True,
                                         append_images=self.sds_training_images[1:],
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

        return text_embeddings

    def denoiser_forward(self, latents, t, text_embeddings):
        return self.denoiser(latents.to(torch.float32),
                             t.to(torch.float32),
                             encoder_hidden_states=text_embeddings.to(torch.float32)).sample.to(torch.float32)
