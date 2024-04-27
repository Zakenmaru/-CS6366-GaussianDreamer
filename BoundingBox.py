# AABB BoundingBox for 3d object
class BoundingBox:
    def __init__(self, points):
        self.dim = len(points[0])
        self.min_bound = [min(p[i] for p in points) for i in range(self.dim)]
        self.max_bound = [max(p[i] for p in points) for i in range(self.dim)]