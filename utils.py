"""
Utility functions for EKF visualization and covariance calculation

Includes tools for:
- Plotting tracking results
- Calculating measurement noise from data
- Analyzing filter performance
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json


def plot_tracking_comparison(timestamps: np.ndarray,
                            true_positions: np.ndarray,
                            raw_measurements: np.ndarray,
                            filtered_positions: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Create comprehensive comparison plots of tracking performance.
    
    Args:
        timestamps: Time array
        true_positions: Ground truth positions (if available)
        raw_measurements: Raw noisy measurements
        filtered_positions: EKF filtered positions
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 3D trajectory
    ax = axes[0, 0]
    ax.plot(raw_measurements[:, 0], raw_measurements[:, 1], 
            'r.', alpha=0.3, label='Raw', markersize=2)
    ax.plot(filtered_positions[:, 0], filtered_positions[:, 1], 
            'b-', label='Filtered', linewidth=2)
    if true_positions is not None:
        ax.plot(true_positions[:, 0], true_positions[:, 1], 
                'g--', label='True', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position error over time
    ax = axes[0, 1]
    if true_positions is not None:
        raw_error = np.linalg.norm(raw_measurements[:, :3] - true_positions, axis=1)
        filtered_error = np.linalg.norm(filtered_positions - true_positions, axis=1)
        ax.plot(timestamps, raw_error * 100, 'r-', alpha=0.5, label='Raw')
        ax.plot(timestamps, filtered_error * 100, 'b-', label='Filtered')
        ax.set_ylabel('Error (cm)')
    else:
        diff = np.linalg.norm(filtered_positions - raw_measurements[:, :3], axis=1)
        ax.plot(timestamps, diff * 100, 'b-', label='Raw-Filtered Difference')
        ax.set_ylabel('Difference (cm)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Position Error/Difference')
    ax.legend()
    ax.grid(True)
    
    # X, Y, Z components
    ax = axes[1, 0]
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax.plot(timestamps, filtered_positions[:, i], label=label, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Components')
    ax.legend()
    ax.grid(True)
    
    # Noise/smoothness
    ax = axes[1, 1]
    raw_diff = np.diff(raw_measurements[:, :3], axis=0)
    filtered_diff = np.diff(filtered_positions, axis=0)
    raw_jerk = np.linalg.norm(raw_diff, axis=1)
    filtered_jerk = np.linalg.norm(filtered_diff, axis=1)
    ax.plot(timestamps[:-1], raw_jerk, 'r-', alpha=0.5, label='Raw')
    ax.plot(timestamps[:-1], filtered_jerk, 'b-', label='Filtered')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Change Between Frames (m)')
    ax.set_title('Smoothness (Lower = Smoother)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def calculate_measurement_covariance(measurement_file: str) -> np.ndarray:
    """
    Calculate measurement covariance matrix from static target data.
    
    Expected file format (JSON):
    {
        "measurements": [
            [x1, y1, z1, yaw1],
            [x2, y2, z2, yaw2],
            ...
        ]
    }
    
    Args:
        measurement_file: Path to JSON file with repeated measurements
        
    Returns:
        R: 4x4 measurement covariance matrix
    """
    with open(measurement_file, 'r') as f:
        data = json.load(f)
    
    measurements = np.array(data['measurements'])
    
    # Calculate covariance
    R = np.cov(measurements.T)
    
    print("Measurement Covariance Matrix (R):")
    print(R)
    print(f"\nStandard deviations:")
    print(f"  X:   {np.sqrt(R[0,0]):.4f} m")
    print(f"  Y:   {np.sqrt(R[1,1]):.4f} m")
    print(f"  Z:   {np.sqrt(R[2,2]):.4f} m")
    print(f"  Yaw: {np.sqrt(R[3,3]):.4f} rad ({np.degrees(np.sqrt(R[3,3])):.2f}°)")
    
    return R


def analyze_filter_performance(timestamps: np.ndarray,
                               true_positions: np.ndarray,
                               raw_measurements: np.ndarray,
                               filtered_positions: np.ndarray) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Returns:
        metrics: Dictionary of performance metrics
    """
    raw_errors = np.linalg.norm(raw_measurements[:, :3] - true_positions, axis=1)
    filtered_errors = np.linalg.norm(filtered_positions - true_positions, axis=1)
    
    metrics = {
        'raw_rmse': np.sqrt(np.mean(raw_errors**2)),
        'filtered_rmse': np.sqrt(np.mean(filtered_errors**2)),
        'raw_max_error': np.max(raw_errors),
        'filtered_max_error': np.max(filtered_errors),
        'raw_std': np.std(raw_errors),
        'filtered_std': np.std(filtered_errors),
        'improvement_percent': (1 - np.mean(filtered_errors) / np.mean(raw_errors)) * 100
    }
    
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Raw RMSE:          {metrics['raw_rmse']*100:.2f} cm")
    print(f"Filtered RMSE:     {metrics['filtered_rmse']*100:.2f} cm")
    print(f"Improvement:       {metrics['improvement_percent']:.1f}%")
    print(f"\nRaw Max Error:     {metrics['raw_max_error']*100:.2f} cm")
    print(f"Filtered Max Error: {metrics['filtered_max_error']*100:.2f} cm")
    print(f"\nRaw Std Dev:       {metrics['raw_std']*100:.2f} cm")
    print(f"Filtered Std Dev:  {metrics['filtered_std']*100:.2f} cm")
    
    return metrics


def plot_uncertainty_ellipse(ekf, ax, n_std: float = 2.0, **kwargs):
    """
    Plot uncertainty ellipse for the current EKF state.
    
    Args:
        ekf: ArmorPlateEKF instance
        ax: Matplotlib axis
        n_std: Number of standard deviations for ellipse
        **kwargs: Additional plot arguments
    """
    from matplotlib.patches import Ellipse
    
    pos, _, _ = ekf.get_state()
    P = ekf.P[0:2, 0:2]  # X-Y covariance
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P)
    
    # Calculate ellipse parameters
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    ellipse = Ellipse(pos[0:2], width, height, angle=angle, 
                     fill=False, edgecolor='b', linewidth=2, **kwargs)
    ax.add_patch(ellipse)


def save_ekf_parameters(ekf, filepath: str):
    """
    Save EKF parameters (Q, R matrices) to a JSON file.
    
    Args:
        ekf: ArmorPlateEKF instance
        filepath: Output file path
    """
    params = {
        'Q': ekf.Q.tolist(),
        'R': ekf.R.tolist(),
        'dt': ekf.dt
    }
    
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"EKF parameters saved to: {filepath}")


def load_ekf_parameters(ekf, filepath: str):
    """
    Load EKF parameters from a JSON file.
    
    Args:
        ekf: ArmorPlateEKF instance to update
        filepath: Input file path
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    
    ekf.Q = np.array(params['Q'])
    ekf.R = np.array(params['R'])
    ekf.dt = params['dt']
    
    print(f"EKF parameters loaded from: {filepath}")


def create_measurement_json_template(output_path: str):
    """
    Create a template JSON file for measurement data collection.
    
    Args:
        output_path: Path to save the template
    """
    template = {
        "description": "Repeated measurements of a stationary target for covariance calculation",
        "target_info": {
            "true_position": [0.0, 0.0, 0.0],
            "true_yaw": 0.0
        },
        "measurements": [
            # Add your measurements here in format: [x, y, z, yaw]
            # Example:
            # [1.523, 0.456, 0.012, 0.123],
            # [1.518, 0.461, 0.009, 0.119],
            # ...
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Measurement template created: {output_path}")
    print("Fill in the 'measurements' array with your SolvePnP results")


def plot_covariance_evolution(timestamps: np.ndarray, 
                              covariance_history: List[np.ndarray]):
    """
    Plot how state uncertainty evolves over time.
    
    Args:
        timestamps: Time array
        covariance_history: List of covariance matrices over time
    """
    # Extract diagonal elements (variances)
    variances = np.array([np.diag(P) for P in covariance_history])
    std_devs = np.sqrt(variances)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position uncertainties
    ax = axes[0]
    ax.plot(timestamps, std_devs[:, 0] * 100, label='X')
    ax.plot(timestamps, std_devs[:, 1] * 100, label='Y')
    ax.plot(timestamps, std_devs[:, 2] * 100, label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Std Dev (cm)')
    ax.set_title('Position Uncertainty Over Time')
    ax.legend()
    ax.grid(True)
    
    # Velocity uncertainties
    ax = axes[1]
    ax.plot(timestamps, std_devs[:, 3], label='vX')
    ax.plot(timestamps, std_devs[:, 4], label='vY')
    ax.plot(timestamps, std_devs[:, 5], label='vZ')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Std Dev (m/s)')
    ax.set_title('Velocity Uncertainty Over Time')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("EKF Utilities Module")
    print("\nAvailable functions:")
    print("  - plot_tracking_comparison()")
    print("  - calculate_measurement_covariance()")
    print("  - analyze_filter_performance()")
    print("  - plot_uncertainty_ellipse()")
    print("  - save_ekf_parameters()")
    print("  - load_ekf_parameters()")
    print("  - create_measurement_json_template()")
    print("  - plot_covariance_evolution()")
    print("\nImport this module to use these utilities in your code.")
    
    # Create a template file
    template_path = r'c:\Uni\Purdue 2nd semester\VIP\robomaster-ekf-tracker\measurement_template.json'
    create_measurement_json_template(template_path)
