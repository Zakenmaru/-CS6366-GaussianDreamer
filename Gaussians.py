# class to store 3D Gaussians as tensor of all gaussians for the image
class Gaussians:
    def __init__(self, positions, colors, alphas, scales, rotations):
        # [x, y, z]
        self.positions = positions
        # [r, g, b]
        self.colors = colors
        # [opacity]
        self.alphas = alphas
        # [scale matrix]
        self.scales = scales
        # [rotation matrix]
        self.rotations = rotations