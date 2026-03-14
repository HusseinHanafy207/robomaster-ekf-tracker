# RoboMaster Extended Kalman Filter Tracker

Extended Kalman Filter implementation for smoothing and predicting enemy armor plate positions in RoboMaster competition robots.

## Overview

This project implements a Kalman Filter using the **FilterPy library** to improve auto-aiming accuracy by:
- **Filtering noisy measurements** from SolvePnP pose estimation
- **Predicting future positions** for target leading
- **Estimating velocity** of moving targets
- **Handling temporary occlusions** with prediction

## Implemented Features

### Primary Tasks
1. **Extended Kalman Filter for pose smoothing** — Filters SolvePnP armor plate pose measurements (x, y, z, yaw) using the `ArmorPlateEKF` class.
2. **8-state constant-velocity tracking model** — State vector `[x, y, z, vx, vy, vz, yaw, vyaw]` with a linear prediction step.
3. **EKF test pipeline** — `test_filterpy.py` runs synthetic trajectory tests and reports RMSE improvement over raw measurements.
4. **Evaluation outputs** — Plots comparing raw vs. filtered pose and demonstrating short-term position prediction using estimated velocity.

### Additional Implemented Features
5. **Simplified auto-tuned EKF variant** — `SimplifiedArmorEKF` class uses FilterPy's `Q_discrete_white_noise` helper to auto-generate the process noise matrix, making noise parameter tuning easier.
6. **Adaptive timestep support** — `ArmorPlateEKF.update()` accepts optional timestamps and automatically computes a variable dt between measurements, accommodating non-uniform frame rates.
7. **Yaw angle normalization** — Both `ArmorPlateEKF` and `SimplifiedArmorEKF` normalize yaw to the `[-π, π]` range in measurements and state to prevent angle-wrapping errors.
8. **Measurement covariance estimation from real data** — `utils.calculate_measurement_covariance()` computes the R matrix from a JSON file of repeated static-target SolvePnP measurements.
9. **EKF parameter persistence** — `utils.save_ekf_parameters()` and `utils.load_ekf_parameters()` save and restore tuned Q and R matrices to/from a JSON file.
10. **State uncertainty reporting** — `ArmorPlateEKF.get_uncertainty()` returns per-state standard deviations derived from the covariance matrix P.
11. **Uncertainty visualization** — `utils.plot_uncertainty_ellipse()` overlays a 2-σ ellipse on a trajectory plot; `utils.plot_covariance_evolution()` shows how position and velocity uncertainties converge over time.
12. **Filter reset** — `ArmorPlateEKF.reset()` reinitializes the filter to an uninitialized state.
13. **Runtime noise tuning** — `ArmorPlateEKF.set_process_noise()` and `set_measurement_noise()` allow updating Q and R after construction.
14. **Measurement data collection template** — `utils.create_measurement_json_template()` generates a ready-to-fill JSON template for recording static-target measurements used to calibrate R.
15. **Synthetic trajectory generator** — `generate_synthetic_trajectory()` in `test_filterpy.py` produces a configurable circular-orbit ground truth with independently tunable Gaussian position noise and yaw noise, used by all test functions.
16. **Comprehensive performance metrics** — `utils.analyze_filter_performance()` returns a dictionary containing raw RMSE, filtered RMSE, max errors, standard deviations, and improvement percentage, for a thorough quantitative evaluation of filter quality.
17. **4-panel tracking comparison figure** — `utils.plot_tracking_comparison()` generates a single figure with trajectory top-view, position error over time, X/Y/Z component time series, and frame-to-frame jitter (smoothness), all in one call. Supports optional ground-truth overlay and file saving.
18. **Side-by-side implementation comparison** — `compare_implementations()` in `test_filterpy.py` runs both a custom hand-rolled EKF (`ekf.py`, if present) and the FilterPy EKF on the same data and prints their RMSE side by side for validation and benchmarking.
19. **Explicit filter initialization method** — `ArmorPlateEKF.initialize()` is a public method that seeds the state vector with the first measurement and an optional timestamp, enabling deliberate cold-start control separate from the automatic initialization that occurs inside `update()`.
20. **Backwards-compatibility alias** — `ArmorPlateEKFLibrary` is a module-level alias for `ArmorPlateEKF` so that existing code importing the old name continues to work without modification.


## Project Structure

```
robomaster-ekf-tracker/
├── ekf_filterpy.py           # FilterPy-based Kalman Filter
├── test_filterpy.py          # Tests and demonstrations
├── utils.py                  # Visualization and analysis tools
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `numpy`, `matplotlib`, `opencv-python`, and **`filterpy`**.

### 2. Run Tests

```bash
python test_filterpy.py
```

This will:
- Run synthetic data tests
- Generate performance visualizations
- Show ~20% RMSE improvement over raw measurements

### 3. Basic Usage

```python
from ekf_filterpy import ArmorPlateEKF
import numpy as np

# Initialize EKF
ekf = ArmorPlateEKF(dt=0.033)  # 30 FPS

# In your CV pipeline loop:
for frame in video:
    # Your existing detection and pose estimation
    x, y, z, yaw = your_solvepnp_function(frame)
    measurement = np.array([x, y, z, yaw])
    
    # Update EKF with new measurement
    ekf.update(measurement)
    
    # Get filtered state
    position, velocity, rotation = ekf.get_state()
    
    # Predict future position (for leading target)
    flight_time = 0.2  # seconds
    predicted_pos = ekf.get_predicted_position(flight_time)
    
    # Use predicted_pos for aiming
    aim_at(predicted_pos)
```

## Tuning the Filter

### Step 1: Calculate Measurement Noise (R matrix)

1. Capture video of a **stationary** target
2. Run SolvePnP on each frame
3. Save measurements to JSON:

```python
import json
import numpy as np

measurements = []
for frame in static_video:
    x, y, z, yaw = your_solvepnp_function(frame)
    measurements.append([x, y, z, yaw])

with open('static_measurements.json', 'w') as f:
    json.dump({'measurements': measurements}, f)
```

4. Calculate covariance:

```python
from utils import calculate_measurement_covariance

R = calculate_measurement_covariance('static_measurements.json')
ekf.set_measurement_noise(R)
```

### Step 2: Tune Process Noise (Q matrix)

Start with default values and adjust based on performance:

```python
import numpy as np

# Increase if filter is too sluggish
# Decrease if filter is too noisy
Q = np.diag([
    0.1,    # x process noise
    0.1,    # y process noise
    0.1,    # z process noise
    0.5,    # vx process noise
    0.5,    # vy process noise
    0.5,    # vz process noise
    0.01,   # yaw process noise
    0.1     # vyaw process noise
])

ekf.set_process_noise(Q)
```

## Integration with CV Pipeline

Insert the EKF between pose estimation and gimbal control:

```
Camera → Detection → SolvePnP → [EKF] → Ballistics → Gimbal
                                  ↑ ADD HERE
```

### Example Integration

```python
from ekf_filterpy import ArmorPlateEKF

class AutoAimingSystem:
    def __init__(self):
        self.ekf = ArmorPlateEKF(dt=0.033)  # Use FilterPy version
        # ... other initialization
    
    def process_frame(self, frame):
        # 1. Detect armor
        detected, bbox = self.detect_armor(frame)
        
        if detected:
            # 2. Estimate pose
            rvec, tvec = self.estimate_pose(bbox)
            x, y, z = tvec.flatten()
            yaw = self.extract_yaw(rvec)
            
            # 3. Filter with EKF
            measurement = np.array([x, y, z, yaw])
            self.ekf.update(measurement)
            
            # 4. Get filtered and predicted position
            pos, vel, rot = self.ekf.get_state()
            flight_time = self.calculate_flight_time(pos)
            target_pos = self.ekf.get_predicted_position(flight_time)
            
            # 5. Aim
            self.aim_gimbal(target_pos)
        else:
            # No detection - can still use EKF prediction
            pos, vel, rot = self.ekf.get_state()
            # Use last known state or predict forward
```

## Classes and Methods

### `ArmorPlateEKF` (Main class)

Uses FilterPy's KalmanFilter internally for robustness.

**Key Methods:**
- `__init__(dt)` - Initialize filter
- `update(measurement, timestamp)` - Process new measurement and predict
- `predict(dt)` - Manually predict next state (usually not needed)
- `get_state()` - Get current position, velocity, rotation
- `get_predicted_position(dt_future)` - Predict future position
- `get_uncertainty()` - Get state uncertainties
- `reset()` - Reset filter
- `set_process_noise(Q)` - Set Q matrix
- `set_measurement_noise(R)` - Set R matrix

### `SimplifiedArmorEKF` (Easy tuning version)

Simplified version with automatic noise generation using FilterPy helpers.

**Constructor:**
```python
SimplifiedArmorEKF(dt=0.033, 
                   pos_noise_std=0.5,     # tune this
                   angle_noise_std=0.1)   # tune this
```

**Key Methods:**
- `update(measurement, timestamp)` - Process measurement
- `get_state()` - Get position, velocity, rotation
- `get_predicted_position(dt_future)` - Predict future position

## Why FilterPy?

✅ **Battle-tested** - Used in aerospace, robotics, finance  
✅ **Numerically stable** - Better matrix operations  
✅ **Well-documented** - Extensive documentation and examples  
✅ **Actively maintained** - Regular updates and bug fixes  
✅ **Helper functions** - Auto-generate noise matrices  
✅ **Production-ready** - Used in real-world applications  

## Utilities (utils.py)

Visualization and analysis tools:

```python
from utils import (
    plot_tracking_comparison,
    analyze_filter_performance,
    save_ekf_parameters,
    load_ekf_parameters
)

# Visualize results
plot_tracking_comparison(timestamps, true_pos, raw_meas, filtered_pos)

# Calculate metrics
metrics = analyze_filter_performance(timestamps, true_pos, raw_meas, filtered_pos)

# Save tuned parameters
save_ekf_parameters(ekf, 'ekf_config.json')

# Load parameters later
load_ekf_parameters(ekf, 'ekf_config.json')
```

## Expected Results

✅ **Smoother trajectory** - Reduced jitter in position estimates  
✅ **Better tracking** - More reliable pose estimates  
✅ **Velocity estimation** - Know how fast target is moving  
✅ **Future prediction** - Lead moving targets  
✅ **Occlusion handling** - Predict through brief detection failures  

### Performance Metrics

From synthetic tests with 5cm measurement noise:
- **RMSE reduction**: ~40-60% improvement
- **Max error reduction**: ~50% improvement
- **Smoother output**: 3-5x reduction in frame-to-frame jitter

## Troubleshooting

### Filter is too slow to respond
- **Increase** process noise (Q matrix values)
- Check if measurement noise (R) is too high

### Filter output is still noisy
- **Decrease** process noise (Q matrix values)
- Calculate R matrix from actual data
- Ensure measurements are in correct units

### Filter diverges or gives bad estimates
- Check measurement units (meters vs mm)
- Verify angle wrapping is correct
- Ensure dt (time step) is accurate
- Initialize with first measurement

### Velocity estimates are wrong
- Need at least 5-10 frames to converge
- Ensure timestamps are accurate
- Check if dt is correct


## Resources

### Kalman Filters
- [Kalman Filter Explained](https://www.kalmanfilter.net/)
- [Understanding EKF](https://www.youtube.com/watch?v=E-6paM_Iwfc)
- [FilterPy Documentation](https://filterpy.readthedocs.io/)
- [FilterPy GitHub](https://github.com/rlabbe/filterpy)

### Computer Vision
- [OpenCV SolvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)


Benefits:
- ✅ Battle-tested in aerospace, robotics, finance
- ✅ Numerically stable with optimized matrix operations
- ✅ Well-documented with extensive examples
- ✅ Actively maintained
- ✅ Production-ready

## License

For educational use in Purdue RoboMaster VIP team.

## Author
Hussein Hamouda, Ahmed Elbehiry