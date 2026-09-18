import cv2
import numpy as np
import random
import math
from pathlib import Path

# -------------------- Kalman Filter for Position Estimation --------------------

class KalmanFilter:
    def __init__(self, dt, std_acc, std_meas):
        """
        dt: Time step (s)
        std_acc: Process noise standard deviation
        std_meas: Measurement noise standard deviation
        """
        self.dt = dt
        
        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1)) 
        
        # State transition model
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control matrix (not used)
        self.B = np.array([[0.5 * dt**2], [0.5 * dt**2], [dt], [dt]])

        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance
        self.Q = np.eye(4) * std_acc**2

        # Measurement noise covariance
        self.R = np.eye(2) * std_meas**2

        # Covariance matrix (initial uncertainty)
        self.P = np.eye(4) * 1000  

    def predict(self):
        """Predicts the next state"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """Updates the state with a new measurement"""
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

    def get_estimated_position(self):
        """Returns estimated position (x, y)"""
        return self.x[:2].flatten()

# ---------------------- RRT* Path Planning ----------------------

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTStar:
    def __init__(self, map_path, start, goal, max_iter=5000, step_size=10, goal_radius=80, neighbor_radius=100):
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map is None:
            raise FileNotFoundError(f"Error: Could not load {map_path}. Check file path and format.")

        self.start = Node(*start)
        self.goal = Node(*goal)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.neighbor_radius = neighbor_radius
        self.nodes = [self.start]
        self.height, self.width = self.map.shape
        self.obstacle_threshold = 80

    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def nearest_node(self, random_node):
        return min(self.nodes, key=lambda node: self.distance(node, random_node))

    def steer(self, from_node, to_node):
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = int(from_node.x + self.step_size * math.cos(theta))
        new_y = int(from_node.y + self.step_size * math.sin(theta))

        new_x = max(0, min(self.width - 1, new_x))
        new_y = max(0, min(self.height - 1, new_y))

        return Node(new_x, new_y)

    def is_collision_free(self, from_node, to_node):
        x_vals = np.linspace(from_node.x, to_node.x, num=20, dtype=int)
        y_vals = np.linspace(from_node.y, to_node.y, num=20, dtype=int)

        for x, y in zip(x_vals, y_vals):
            if 0 <= y < self.height and 0 <= x < self.width:
                if self.map[y, x] < self.obstacle_threshold:
                    return False
        return True

    def generate_path(self):
        path = []
        current_node = self.goal
        while current_node and current_node.parent:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent

        if not path:
            print("⚠️ No valid path found!")
            return None

        path.append((self.start.x, self.start.y))
        return path[::-1]

    def run(self, kalman_filter):
        for i in range(self.max_iter):
            estimated_position = kalman_filter.get_estimated_position()
            rand_x = random.randint(0, self.width - 1)
            rand_y = random.randint(0, self.height - 1)
            random_node = Node(rand_x, rand_y)
            nearest = self.nearest_node(random_node)
            new_node = self.steer(nearest, random_node)

            if self.is_collision_free(nearest, new_node):
                new_node.parent = nearest
                self.nodes.append(new_node)

                if self.distance(new_node, self.goal) < self.goal_radius:
                    self.goal = new_node
                    print(f"Path found in {i+1} iterations!")
                    return self.generate_path()

        print("❌ No path found.")
        return None

# ---------------------- Hybrid Execution ----------------------

# Load the map
PROJECT_ROOT = Path(__file__).resolve().parents[2]
map_path = str(PROJECT_ROOT / "assets" / "sample_maps" / "mar8.jpg")
start_point = (70, 20)
goal_point = (100, 100)

# Initialize Kalman Filter
kf = KalmanFilter(dt=1, std_acc=1, std_meas=10)

# Simulated noisy measurements
measurements = np.array([[500, 300], [405, 310], [410, 320], [420, 330]])

# Process measurements through Kalman Filter
for z in measurements:
    kf.predict()
    kf.update(np.array(z).reshape(2, 1))
    print("Estimated Position (after Kalman Filter):", kf.get_estimated_position())

# Run RRT* using Kalman Filter estimated position
rrt_star = RRTStar(map_path, start=kf.get_estimated_position().astype(int), goal=goal_point)
path = rrt_star.run(kf)

# Display result
if path:
    img = cv2.imread(map_path)
    for i in range(len(path) - 1):
        cv2.line(img, path[i], path[i + 1], (255, 0, 0), 2)

    cv2.imshow("Drone Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
