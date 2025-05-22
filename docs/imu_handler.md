# Module src.imu_handler

> IMU (Inertial Measurement Unit) Handler for DepthAI.
>
> This module defines the `ImuHandler` class, which is responsible for configuring
> and managing the IMU sensor on a DepthAI device. It handles setting up the
> IMU node within the DepthAI pipeline, fetching raw sensor data (accelerometer and
> gyroscope), performing gyroscope calibration to account for bias, and calculating
> a basic orientation estimate through direct integration of gyroscope readings.

---
## Class ImuHandler

Manages IMU sensor interactions on the DepthAI device.

This includes setup, data fetching, gyroscope calibration, and basic orientation calculation.
The orientation calculation is a simple integration of gyroscope data and is prone to drift
over time. For more robust orientation, advanced sensor fusion algorithms (e.g., Kalman filter,
Madgwick, Mahony) would be required.

#### Attributes
*   **pipeline** (*dai.Pipeline*): The DepthAI pipeline object to which the IMU node is added.
*   **imu_node** (*dai.node.IMU | None*): The DepthAI IMU node instance.
*   **xout_imu** (*dai.node.XLinkOut | None*): The XLinkOut node for IMU data.
*   **base_ts** (*Optional[float]*): Base timestamp for synchronizing IMU packet timestamps. Initialized upon receiving the first packet.
*   **reference_orientation_rad** (*np.ndarray*): Current estimated orientation (roll, pitch, yaw) in radians. This is updated by integrating gyroscope data.
*   **gyro_calibrated** (*bool*): Flag indicating whether gyroscope calibration has been performed.
*   **gyro_bias** (*np.ndarray*): Estimated gyroscope bias (rad/s) for X, Y, Z axes. Calculated during calibration.

### Methods
*   `__init__`
*   `_setup_imu_node`
*   `get_output_queues`
*   `time_delta_to_millis`
*   `calibrate_gyroscope`
*   `reset_orientation`
*   `process_imu_packet`

---
### `__init__`
```python
def __init__(self, pipeline: dai.Pipeline, report_threshold: int = 1, max_reports: int = 10, accel_hz: int = 500, gyro_hz: int = 400):
```
Initializes the IMUHandler and sets up the IMU node in the pipeline.

#### Arguments
*   **pipeline** (*dai.Pipeline*): The DepthAI pipeline object.
*   **report_threshold** (*int, optional*): Number of IMU packets to batch on the device before sending a report to the host. Defaults to 1.
*   **max_reports** (*int, optional*): Maximum number of batched reports the device should keep. Defaults to 10.
*   **accel_hz** (*int, optional*): Desired sampling rate (frequency in Hz) for the accelerometer. Common values are 100, 200, 500. Defaults to 500.
*   **gyro_hz** (*int, optional*): Desired sampling rate (frequency in Hz) for the gyroscope. Common values are 100, 200, 400. Defaults to 400.

---
### `_setup_imu_node`
```python
def _setup_imu_node(self, report_threshold: int, max_reports: int, accel_hz: int, gyro_hz: int) -> None:
```
Creates and configures the IMU node in the pipeline.

This private method is called during initialization. It sets up the IMU sensor types
(accelerometer and gyroscope), their reporting frequency, and batching parameters.

#### Arguments
*   **report_threshold** (*int*): Number of packets to batch before sending a report.
*   **max_reports** (*int*): Maximum number of batched reports to keep.
*   **accel_hz** (*int*): Sampling rate for accelerometer.
*   **gyro_hz** (*int*): Sampling rate for gyroscope.

---
### `get_output_queues`
```python
def get_output_queues(self, device: dai.Device) -> Dict[str, dai.DataOutputQueue]:
```
Gets the output queue for IMU data from the DepthAI device.

#### Arguments
*   **device** (*dai.Device*): The DepthAI device instance.

#### Returns
*   (*Dict[str, dai.DataOutputQueue]*): A dictionary containing the IMU data output queue, keyed by "imu".

---
### `time_delta_to_millis`
```python
@staticmethod
def time_delta_to_millis(delta: 'np.timedelta64') -> float:
```
Converts a NumPy timedelta64 object to milliseconds.

#### Arguments
*   **delta** (*np.timedelta64*): The timedelta object to convert.

#### Returns
*   (*float*): The equivalent duration in milliseconds.

---
### `calibrate_gyroscope`
```python
def calibrate_gyroscope(self, imu_queue: dai.DataOutputQueue, num_samples: int = 200, wait_time_ms: int = 5) -> None:
```
Calibrates the gyroscope by averaging readings to determine bias.

The device should be kept stationary during calibration. This method collects
a specified number of gyroscope readings and calculates their mean, which is
then stored as the gyroscope bias. This bias is subtracted from subsequent
gyroscope readings in `process_imu_packet`.

#### Arguments
*   **imu_queue** (*dai.DataOutputQueue*): The IMU data queue to read from. This should be a blocking queue for effective calibration.
*   **num_samples** (*int, optional*): Number of gyroscope samples to average for calibration. Defaults to 200.
*   **wait_time_ms** (*int, optional*): Time in milliseconds to wait between polling the queue if it's empty. This is less relevant for a blocking queue but serves as a fallback. Defaults to 5.

---
### `reset_orientation`
```python
def reset_orientation(self) -> None:
```
Resets the current estimated orientation to zero (roll, pitch, yaw = 0,0,0).
Also resets the base timestamp for calculations.

---
### `process_imu_packet`
```python
def process_imu_packet(self, imu_packet: dai.IMUPacket, last_gyro_ts_device: Optional[np.datetime64] = None) -> Optional[Dict[str, any]]:
```
Processes a single IMU packet to extract sensor data and update orientation.

This method extracts accelerometer and gyroscope readings, applies gyroscope bias
correction if calibrated, and integrates gyroscope data to update the
`reference_orientation_rad`. The orientation is a simple Euler angle integration
and will drift over time.

#### Arguments
*   **imu_packet** (*dai.IMUPacket*): A single IMU packet from the DepthAI device.
*   **last_gyro_ts_device** (*Optional[np.datetime64], optional*): Timestamp of the previous gyroscope reading from the device. Used to calculate the time delta (`dt`) for integration. If None, `dt` will be zero for the first packet. Defaults to None.

#### Returns
*   (*Optional[Dict[str, any]]*): A dictionary containing processed IMU data:
    *   "accel_ts_ms" (float): Accelerometer timestamp in milliseconds relative to base.
    *   "accel_m_s2" (np.ndarray): Accelerometer readings (X,Y,Z) in m/s^2.
    *   "gyro_ts_ms" (float): Gyroscope timestamp in milliseconds relative to base.
    *   "gyro_rad_s" (np.ndarray): Gyroscope readings (X,Y,Z) in rad/s, bias corrected.
    *   "orientation_deg" (np.ndarray): Estimated orientation (roll, pitch, yaw) in degrees.
    *   "raw_gyro_rad_s" (np.ndarray): Raw gyroscope readings (X,Y,Z) in rad/s, uncorrected.
    *   "new_last_gyro_ts_device" (np.datetime64): Current gyroscope timestamp, to be passed as `last_gyro_ts_device` in the next call.
    Returns `None` if the input `imu_packet` is None or essential data is missing.

---
```
