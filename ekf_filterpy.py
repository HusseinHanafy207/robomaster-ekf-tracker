"""
Extended Kalman Filter using FilterPy library for RoboMaster Armor Plate Tracking

This implementation uses the robust FilterPy library.
FilterPy is the standard Python library for Kalman filtering.

State Vector: [x, y, z, vx, vy, vz, yaw, vyaw]
- x, y, z: 3D position in meters
- vx, vy, vz: velocity in m/s
- yaw: rotation angle in radians
- vyaw: angular velocity in rad/s
"""

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, Optional


class ArmorPlateEKF:
  
    
    def __init__(self, dt: float = 0.033):
        """
        Initialize the Kalman Filter using FilterPy.
        
        Args:
            dt: Time step between measurements (default: 0.033s ≈ 30fps)
        """
        self.dt = dt
        
        # State dimension: [x, y, z, vx, vy, vz, yaw, vyaw]
        dim_x = 8
        # Measurement dimension: [x, y, z, yaw]
        dim_z = 4
        
        # Create FilterPy ExtendedKalmanFilter as per project requirements
        self.kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initial state (will be set on first measurement)
        self.kf.x = np.zeros(dim_x)
        
        # Initial covariance (high uncertainty)
        self.kf.P = np.eye(dim_x) * 1000.0
        
        # State transition function (constant velocity model)
        self.kf.F = self._get_state_transition_matrix(dt)
        
        # Process noise covariance (Q)
        # Models uncertainty in the motion model
        self.kf.Q = np.diag([
            0.1,    # x process noise
            0.1,    # y process noise
            0.1,    # z process noise
            0.5,    # vx process noise
            0.5,    # vy process noise
            0.5,    # vz process noise
            0.01,   # yaw process noise
            0.1     # vyaw process noise
        ])
        
        # Measurement noise covariance (R)
        # Models uncertainty in SolvePnP measurements
        self.kf.R = np.diag([
            0.05,   # x measurement noise (meters)
            0.05,   # y measurement noise
            0.05,   # z measurement noise
            0.1     # yaw measurement noise (radians)
        ])
        
        # Initialization flag
        self.is_initialized = False
        self.last_time = None
        
    def _get_state_transition_matrix(self, dt: float) -> np.ndarray:
    
        F = np.eye(8)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        F[6, 7] = dt  # yaw += vyaw * dt
        return F
    
    def _hx(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function for EKF.
        Maps state to measurement space: extracts [x, y, z, yaw] from state.
        
        Required by ExtendedKalmanFilter for the update step.
        
        Args:
            x: State vector [x, y, z, vx, vy, vz, yaw, vyaw]
            
        Returns:
            Measurement vector [x, y, z, yaw]
        """
        return np.array([x[0], x[1], x[2], x[6]])
    
    def _HJacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian for EKF.
        Computes the Jacobian of the measurement function.
        
        Since our measurement function is linear (just extracting values),
        this Jacobian is constant regardless of the state.
        
        Required by ExtendedKalmanFilter for the update step.
        
        Args:
            x: State vector (unused, but required by FilterPy)
            
        Returns:
            4x8 measurement Jacobian matrix H
        """
        H = np.zeros((4, 8))
        H[0, 0] = 1.0  # ∂h₁/∂x = 1 (measure x directly)
        H[1, 1] = 1.0  # ∂h₂/∂y = 1 (measure y directly)
        H[2, 2] = 1.0  # ∂h₃/∂z = 1 (measure z directly)
        H[3, 6] = 1.0  # ∂h₄/∂yaw = 1 (measure yaw directly)
        return H
    
    def initialize(self, measurement: np.ndarray, timestamp: Optional[float] = None):
        """
        Initialize the filter with the first measurement.
        
        Args:
            measurement: [x, y, z, yaw] from SolvePnP
            timestamp: Optional timestamp in seconds
        """
        self.kf.x[0:3] = measurement[0:3]  # Position
        self.kf.x[3:6] = 0.0                # Initial velocity = 0
        self.kf.x[6] = measurement[3]       # Yaw
        self.kf.x[7] = 0.0                  # Initial angular velocity = 0
        
        self.is_initialized = True
        self.last_time = timestamp
        
    def predict(self, dt: Optional[float] = None):
        """
        Predict the next state.
        
        Args:
            dt: Time step (uses self.dt if not provided)
        """
        if dt is None:
            dt = self.dt
        
        # Update F matrix if dt changed
        if dt != self.dt:
            self.kf.F = self._get_state_transition_matrix(dt)
        
        # FilterPy handles the prediction
        self.kf.predict()
        
        # Restore default F if we changed it
        if dt != self.dt:
            self.kf.F = self._get_state_transition_matrix(self.dt)
    
    def update(self, measurement: np.ndarray, timestamp: Optional[float] = None):
        """
        Update the state estimate with a new measurement.
        
        Args:
            measurement: [x, y, z, yaw] from SolvePnP
            timestamp: Optional timestamp for adaptive dt
        """
        if not self.is_initialized:
            self.initialize(measurement, timestamp)
            return
        
        # Calculate dt if timestamp provided
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        else:
            dt = self.dt
            if timestamp is not None:
                self.last_time = timestamp
        
        # Predict step
        self.predict(dt)
        
        # Normalize angle in measurement
        measurement_copy = measurement.copy()
        measurement_copy[3] = self._normalize_angle(measurement_copy[3])
        
        # Update step - ExtendedKalmanFilter requires HJacobian and Hx functions
        self.kf.update(measurement_copy, HJacobian=self._HJacobian, Hx=self._hx)
        
        # Normalize yaw in state
        self.kf.x[6] = self._normalize_angle(self.kf.x[6])
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current filtered state estimate.
        
        Returns:
            position: [x, y, z] in meters
            velocity: [vx, vy, vz] in m/s
            rotation: [yaw, vyaw] in radians and rad/s
        """
        position = self.kf.x[0:3]
        velocity = self.kf.x[0:3]
        velocity = self.kf.x[3:6]
        rotation = self.kf.x[6:8]
        return position, velocity, rotation
    
    def get_predicted_position(self, dt_future: float) -> np.ndarray:
        """
        Get predicted position at a future time.
        
        Useful for leading the target in auto-aiming.
        
        Args:
            dt_future: Time into the future (seconds)
            
        Returns:
            predicted_position: [x, y, z] at time t + dt_future
        """
        position = self.kf.x[0:3]
        velocity = self.kf.x[3:6]
        predicted_position = position + velocity * dt_future
        return predicted_position
    
    def get_uncertainty(self) -> np.ndarray:
        """
        Get the current state uncertainty (standard deviations).
        
        Returns:
            std_devs: Standard deviation for each state variable
        """
        return np.sqrt(np.diag(self.kf.P))
    
    def reset(self):
        """Reset the filter to uninitialized state."""
        self.kf.x = np.zeros(8)
        self.s_initialized = False
        self.last_time = None
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def set_process_noise(self, Q: np.ndarray):
        """
        Set custom process noise covariance matrix.
        
        Args:
            Q: 8x8 covariance matrix
        """
        assert Q.shape == (8, 8), "Q must be 8x8"
        self.kf.Q = Q
    
    def set_measurement_noise(self, R: np.ndarray):
        """
        Set custom measurement noise covariance matrix.
        
        Args:
            R: 4x4 covariance matrix
        """
        assert R.shape == (4, 4), "R must be 4x4"
        self.kf.R = R


class SimplifiedArmorEKF:
    """
    Simplified Extended Kalman Filter using FilterPy's helper functions.
    
    This version uses FilterPy's Q_discrete_white_noise to automatically
    generate process noise matrices, which is often easier to tune.
    Uses ExtendedKalmanFilter to meet project requirements.
    """
    
    def __init__(self, dt: float = 0.033, 
                 pos_noise_std: float = 0.5,
                 angle_noise_std: float = 0.1):
        """
        Initialize simplified EKF with automatic noise generation.
        
        Args:
            dt: Time step
            pos_noise_std: Process noise for position/velocity (m/s²)
            angle_noise_std: Process noise for angle/angular velocity (rad/s²)
        """
        self.dt = dt
        dim_x = 8
        dim_z = 4
        
        self.kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.x = np.zeros(dim_x)
        self.kf.P = np.eye(dim_x) * 1000.0
        
        # State transition matrix
        self.kf.F = np.eye(8)
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        self.kf.F[6, 7] = dt
        
        # Use FilterPy's helper to generate process noise
        # Creates noise for position-velocity pairs
        q_pos = Q_discrete_white_noise(dim=2, dt=dt, var=pos_noise_std**2)
        q_angle = Q_discrete_white_noise(dim=2, dt=dt, var=angle_noise_std**2)
        
        # Build block diagonal Q matrix
        self.kf.Q = np.zeros((8, 8))
        self.kf.Q[0:2, 0:2] = q_pos  # x, vx
        self.kf.Q[2:4, 2:4] = q_pos  # y, vy (swap indices)
        self.kf.Q[4:6, 4:6] = q_pos  # z, vz (swap indices)
        self.kf.Q[6:8, 6:8] = q_angle  # yaw, vyaw
        
        # Reorder to match our state ordering [x, y, z, vx, vy, vz, yaw, vyaw]
        # Build properly ordered Q
        self.kf.Q = np.zeros((8, 8))
        # X subsystem
        self.kf.Q[0, 0] = q_pos[0, 0]  # x variance
        self.kf.Q[0, 3] = q_pos[0, 1]  # x-vx covariance
        self.kf.Q[3, 0] = q_pos[1, 0]  # vx-x covariance
        self.kf.Q[3, 3] = q_pos[1, 1]  # vx variance
        # Y subsystem
        self.kf.Q[1, 1] = q_pos[0, 0]
        self.kf.Q[1, 4] = q_pos[0, 1]
        self.kf.Q[4, 1] = q_pos[1, 0]
        self.kf.Q[4, 4] = q_pos[1, 1]
        # Z subsystem
        self.kf.Q[2, 2] = q_pos[0, 0]
        self.kf.Q[2, 5] = q_pos[0, 1]
        self.kf.Q[5, 2] = q_pos[1, 0]
        self.kf.Q[5, 5] = q_pos[1, 1]
        # Yaw subsystem
        self.kf.Q[6, 6] = q_angle[0, 0]
        self.kf.Q[6, 7] = q_angle[0, 1]
        self.kf.Q[7, 6] = q_angle[1, 0]
        self.kf.Q[7, 7] = q_angle[1, 1]
        
        # Measurement noise (same as before)
        self.kf.R = np.diag([0.05, 0.05, 0.05, 0.1])
        
        self.is_initialized = False
        self.last_time = None
    
    def _hx(self, x: np.ndarray) -> np.ndarray:
        """Measurement function for EKF."""
        return np.array([x[0], x[1], x[2], x[6]])
    
    def _HJacobian(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian for EKF."""
        H = np.zeros((4, 8))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 6] = 1.0
        return H
    
    def update(self, measurement: np.ndarray, timestamp: Optional[float] = None):
        """Update with new measurement."""
        if not self.is_initialized:
            self.kf.x[0:3] = measurement[0:3]
            self.kf.x[6] = measurement[3]
            self.is_initialized = True
            self.last_time = timestamp
            return
        
        # Predict and update
        self.kf.predict()
        
        # Normalize measurement angle
        measurement_copy = measurement.copy()
        measurement_copy[3] = self._normalize_angle(measurement_copy[3])
        
        # Update with EKF measurement functions
        self.kf.update(measurement_copy, HJacobian=self._HJacobian, Hx=self._hx)
        
        # Normalize state angle
        self.kf.x[6] = self._normalize_angle(self.kf.x[6])
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current state."""
        return self.kf.x[0:3], self.kf.x[3:6], self.kf.x[6:8]
    
    def get_predicted_position(self, dt_future: float) -> np.ndarray:
        """Get future predicted position."""
        return self.kf.x[0:3] + self.kf.x[3:6] * dt_future
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


# For backwards compatibility, use FilterPy version as default
ArmorPlateEKFLibrary = ArmorPlateEKF


if __name__ == "__main__":
    print("Extended Kalman Filter using FilterPy library")
    print("\nThis implementation uses the robust FilterPy library")
    print("instead of custom matrix operations.")
    print("\nUsage:")
    print("  from ekf_filterpy import ArmorPlateEKF")
    print("  ekf = ArmorPlateEKF(dt=0.033)")
    print("  ekf.update(measurement)")
    print("  position, velocity, rotation = ekf.get_state()")
