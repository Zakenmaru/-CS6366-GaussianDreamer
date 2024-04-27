# TriangleMesh stores all triangles in the form of vertices and colors
class TriangleMesh:
    def __init__(self):
        # in the form of [vertex1, vertex2, vertex3]
        self.vertices = []
        # color for each triangle
        self.colors = []