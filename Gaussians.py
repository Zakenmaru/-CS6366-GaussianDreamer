class Gaussians:
    def __init__(self, positions, colors, covariance, alpha):
        self.positions = positions
        self.colors = colors
        self.covariance = covariance
        self.alpha = alpha