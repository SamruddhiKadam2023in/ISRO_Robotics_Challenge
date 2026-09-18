import numpy as np

class KalmanFilter:
    def __init__(self, dt, u, std_acc, std_meas):
        """
        dt: Time step (s)
        u: External motion (control input, assumed zero here)
        std_acc: Process noise (acceleration noise standard deviation)
        std_meas: Measurement noise standard deviation
        """
        
        # State vector [x, y, vx, vy] (position and velocity)
        self.x = np.zeros((4, 1))
        
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Control matrix (not used in this case)
        self.B = np.array([[0.5 * dt**2],
                           [0.5 * dt**2],
                           [dt],
                           [dt]])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Process noise covariance
        self.Q = np.eye(4) * std_acc**2
        
        # Measurement noise covariance
        self.R = np.eye(2) * std_meas**2
        
        # Covariance matrix (initial uncertainty)
        self.P = np.eye(4) * 1000
        
    def predict(self, u=0):
        """Predicts the next state"""
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, z):
        """Updates the state with a new measurement"""
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
    
    def get_state(self):
        return self.x[:2]  # Return only position (x, y)

# Example usage
dt = 1  # 1 second time step
kf = KalmanFilter(dt, u=0, std_acc=1, std_meas=10)

# Simulated noisy measurements
measurements = np.array([[500, 300], [405, 310], [410, 320], [420, 330]])

for z in measurements:
    kf.predict()
    kf.update(np.array(z).reshape(2, 1))
    print("Estimated position:", kf.get_state().flatten())
