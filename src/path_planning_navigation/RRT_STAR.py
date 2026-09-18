import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = 0  # Path cost from start node

class RRTStar:
    def __init__(self, start, goal, map_size, step_size=10, max_iter=500, search_radius=15):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_size = map_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.obstacles = []

    def add_obstacle(self, x, y, radius):
        """Add circular obstacles."""
        self.obstacles.append((x, y, radius))

    def distance(self, node1, node2):
        """Calculate Euclidean distance."""
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a random point."""
        return min(self.nodes, key=lambda node: self.distance(node, random_point))

    def get_nearby_nodes(self, new_node):
        """Find nodes within a given search radius."""
        return [node for node in self.nodes if self.distance(node, new_node) < self.search_radius]

    def is_collision_free(self, node):
        """Check if a node collides with any obstacles."""
        for ox, oy, r in self.obstacles:
            if self.distance(node, Node(ox, oy)) < r:
                return False
        return True

    def steer(self, nearest, random_point):
        """Steer towards the random point while maintaining step size."""
        theta = np.arctan2(random_point.y - nearest.y, random_point.x - nearest.x)
        new_x = nearest.x + self.step_size * np.cos(theta)
        new_y = nearest.y + self.step_size * np.sin(theta)

        new_node = Node(new_x, new_y, parent=nearest)
        return new_node if self.is_collision_free(new_node) else None

    def choose_best_parent(self, new_node, nearby_nodes):
        """Choose the best parent based on the lowest cost."""
        best_parent = new_node.parent
        min_cost = new_node.parent.cost + self.distance(new_node, new_node.parent)

        for node in nearby_nodes:
            cost = node.cost + self.distance(node, new_node)
            if cost < min_cost:
                best_parent = node
                min_cost = cost

        new_node.parent = best_parent
        new_node.cost = min_cost

    def rewire(self, new_node, nearby_nodes):
        """Rewire nearby nodes if connecting through new_node is beneficial."""
        for node in nearby_nodes:
            new_cost = new_node.cost + self.distance(new_node, node)
            if new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost

    def build_rrt_star(self):
        """Construct the RRT* tree with path optimization."""
        for _ in range(self.max_iter):
            rand_x, rand_y = random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1])
            random_point = Node(rand_x, rand_y)

            nearest = self.get_nearest_node(random_point)
            new_node = self.steer(nearest, random_point)

            if new_node:
                nearby_nodes = self.get_nearby_nodes(new_node)
                if nearby_nodes:
                    self.choose_best_parent(new_node, nearby_nodes)
                    self.rewire(new_node, nearby_nodes)

                self.nodes.append(new_node)

                if self.distance(new_node, self.goal) < self.step_size:
                    print("Goal reached!")
                    return new_node
        return None

    def get_path(self, last_node):
        """Backtrack to get the optimal path."""
        path = []
        node = last_node
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def plot(self, path=None):
        """Plot the RRT* tree and optimized path."""
        plt.figure(figsize=(8, 6))
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])

        # Plot obstacles
        for ox, oy, r in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='red', fill=True)
            plt.gca().add_patch(circle)

        # Plot tree
        for node in self.nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "g.-")

        # Plot path
        if path:
            px, py = zip(*path)
            plt.plot(px, py, "b-", linewidth=2, label="Optimized Path")

        plt.scatter(self.start.x, self.start.y, c="blue", s=100, label="Start")
        plt.scatter(self.goal.x, self.goal.y, c="green", s=100, label="Goal")
        plt.legend()
        plt.grid()
        plt.show()

# ----------------- Run the RRT* Algorithm -----------------
if __name__ == "__main__":
    start_pos = (10, 20)
    goal_pos = (180, 180)
    map_size = (200, 200)  # Map size (width x height)

    rrt_star = RRTStar(start_pos, goal_pos, map_size)
    rrt_star.add_obstacle(100, 100, 20)  # Example obstacle
    rrt_star.add_obstacle(50, 150, 15)
    rrt_star.add_obstacle(120, 50, 25)

    last_node = rrt_star.build_rrt_star()
    if last_node:
        path = rrt_star.get_path(last_node)
        rrt_star.plot(path)
    else:
        print("Failed to find a path.")
