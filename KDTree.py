# Node for the KDtree with its value, left child, right chid, and index
class KDNode:
    def __init__(self, val, left, right, index):
        self.val = val
        self.left = left
        self.right = right
        self.index = index


# KDTree used to obtain closest points
class KDTree:
    def __init__(self, points, dim):
        self.dim = dim
        self.points = []

        # store points and their indices
        self.points = [(p, i) for i, p in enumerate(points)]

        def build_tree(points, depth):
            if not points:
                return None

            points.sort(key=lambda x: x[0][depth % dim])
            m = len(points)//2

            return KDNode(
              val=points[m][0],
              left=build_tree(
                  points=points[:m],
                  depth=depth+1,
              ),
              right=build_tree(
                  points=points[m+1:],
                  depth=depth+1,
              ),
              index=points[m][1]
            )

        self.tree = build_tree(points=self.points, depth=0)

    def get_tree(self):
        return self.tree

    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.val)
            self.inorder(node.right)

    def nearest_neighbor(self, target_pt):
        best_node = None
        best_dist = float('inf')
        def nearest_neighbor_helper(node, depth):
                nonlocal best_dist, best_node
                if not node:
                    return

                dist = sum((node.val[i] - target_pt[i]) ** 2 for i in range(len(target_pt)))

                if dist < best_dist:
                    best_node = node
                    best_dist = dist

                axis = depth % len(target_pt)
                diff = target_pt[axis] - node.val[axis]

                close = node.left
                away = node.right

                if diff > 0:
                    close = node.right
                    away = node.left

                nearest_neighbor_helper(close, depth + 1)

                if diff ** 2 < best_dist:
                    nearest_neighbor_helper(away, depth + 1)
        nearest_neighbor_helper(self.tree, 0)
        return best_node.val, best_node.index