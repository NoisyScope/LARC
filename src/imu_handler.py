"""
IMU (Inertial Measurement Unit) Handler for DepthAI.

This module defines the `ImuHandler` class, which is responsible for configuring
and managing the IMU sensor on a DepthAI device. It handles setting up the
IMU node within the DepthAI pipeline, fetching raw sensor data (accelerometer and
gyroscope), performing gyroscope calibration to account for bias, and calculating
a basic orientation estimate through direct integration of gyroscope readings.
"""
import depthai as dai
import numpy as np
import time # For timeDeltaToMilliS, though it could be in utils
from typing import Dict, Tuple, Optional, List # For type hinting

class ImuHandler:
    """
    Manages IMU sensor interactions on the DepthAI device.

    This includes setup, data fetching, gyroscope calibration, and basic orientation calculation.
    The orientation calculation is a simple integration of gyroscope data and is prone to drift
    over time. For more robust orientation, advanced sensor fusion algorithms (e.g., Kalman filter,
    Madgwick, Mahony) would be required.

    Attributes:
        pipeline (dai.Pipeline): The DepthAI pipeline object to which the IMU node is added.
        imu_node (dai.node.IMU | None): The DepthAI IMU node instance.
        xout_imu (dai.node.XLinkOut | None): The XLinkOut node for IMU data.
        base_ts (Optional[float]): Base timestamp for synchronizing IMU packet timestamps.
            Initialized upon receiving the first packet.
        reference_orientation_rad (np.ndarray): Current estimated orientation (roll, pitch, yaw)
            in radians. This is updated by integrating gyroscope data.
        gyro_calibrated (bool): Flag indicating whether gyroscope calibration has been performed.
        gyro_bias (np.ndarray): Estimated gyroscope bias (rad/s) for X, Y, Z axes.
            Calculated during calibration.
    """
    def __init__(self,
                 pipeline: dai.Pipeline,
                 report_threshold: int = 1,
                 max_reports: int = 10,
                 accel_hz: int = 500,
                 gyro_hz: int = 400):
        """
        Initializes the IMUHandler and sets up the IMU node in the pipeline.

        Args:
            pipeline (dai.Pipeline): The DepthAI pipeline object.
            report_threshold (int): Number of IMU packets to batch on the device before
                sending a report to the host. Defaults to 1.
            max_reports (int): Maximum number of batched reports the device should keep.
                Defaults to 10.
            accel_hz (int): Desired sampling rate (frequency in Hz) for the accelerometer.
                Common values are 100, 200, 500. Defaults to 500.
            gyro_hz (int): Desired sampling rate (frequency in Hz) for the gyroscope.
                Common values are 100, 200, 400. Defaults to 400.
        """
        self.pipeline: dai.Pipeline = pipeline
        self.imu_node: Optional[dai.node.IMU] = None
        self.xout_imu: Optional[dai.node.XLinkOut] = None

        self._setup_imu_node(report_threshold, max_reports, accel_hz, gyro_hz)

        self.base_ts: Optional[np.datetime64] = None # For timestamp synchronization
        self.reference_orientation_rad: np.ndarray = np.array([0.0, 0.0, 0.0]) # Roll, Pitch, Yaw in radians
        self.gyro_calibrated: bool = False
        self.gyro_bias: np.ndarray = np.zeros(3) # Bias for gx, gy, gz (rad/s)

    def _setup_imu_node(self, report_threshold: int, max_reports: int, accel_hz: int, gyro_hz: int) -> None:
        """
        Creates and configures the IMU node in the pipeline.

        This private method is called during initialization. It sets up the IMU sensor types
        (accelerometer and gyroscope), their reporting frequency, and batching parameters.

        Args:
            report_threshold (int): Number of packets to batch before sending a report.
            max_reports (int): Maximum number of batched reports to keep.
            accel_hz (int): Sampling rate for accelerometer.
            gyro_hz (int): Sampling rate for gyroscope.
        """
        self.imu_node = self.pipeline.create(dai.node.IMU)
        # Using GYROSCOPE_CALIBRATED is generally preferred if available and reliable on the device.
        # If issues arise, GYROSCOPE_RAW can be used, but host-side calibration or sensor fusion
        # becomes more critical.
        self.imu_node.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, accel_hz)
        self.imu_node.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, gyro_hz)
        self.imu_node.setBatchReportThreshold(report_threshold)
        self.imu_node.setMaxBatchReports(max_reports) # Configure IMU to send batched reports

        self.xout_imu = self.pipeline.create(dai.node.XLinkOut)
        self.xout_imu.setStreamName("imu")
        self.imu_node.out.link(self.xout_imu.input)

    def get_output_queues(self, device: dai.Device) -> Dict[str, dai.DataOutputQueue]:
        """
        Gets the output queue for IMU data from the DepthAI device.

        Args:
            device (dai.Device): The DepthAI device instance.

        Returns:
            Dict[str, dai.DataOutputQueue]: A dictionary containing the IMU data
            output queue, keyed by "imu".
        """
        queues = {}
        if self.xout_imu:
            # MaxSize influences buffering. Larger values can smooth out processing delays
            # but increase latency.
            queues["imu"] = device.getOutputQueue(name="imu", maxSize=30, blocking=False)
        return queues

    @staticmethod
    def time_delta_to_millis(delta: 'np.timedelta64') -> float:
        """
        Converts a NumPy timedelta64 object to milliseconds.

        Args:
            delta (np.timedelta64): The timedelta object to convert.

        Returns:
            float: The equivalent duration in milliseconds.
        """
        return delta.astype('timedelta64[us]').item() / 1000.0


    def calibrate_gyroscope(self, imu_queue: dai.DataOutputQueue, num_samples: int = 200, wait_time_ms: int = 5) -> None:
        """
        Calibrates the gyroscope by averaging readings to determine bias.

        The device should be kept stationary during calibration. This method collects
        a specified number of gyroscope readings and calculates their mean, which is
        then stored as the gyroscope bias. This bias is subtracted from subsequent
        gyroscope readings in `process_imu_packet`.

        Args:
            imu_queue (dai.DataOutputQueue): The IMU data queue to read from.
                This should be a blocking queue for effective calibration.
            num_samples (int): Number of gyroscope samples to average for calibration.
                Defaults to 200.
            wait_time_ms (int): Time in milliseconds to wait between polling the queue
                if it's empty. This is less relevant for a blocking queue but serves
                as a fallback. Defaults to 5.
        """
        print("Starting gyroscope calibration. Keep the device still...")
        gyro_readings_list: List[np.ndarray] = [] # Explicitly list of numpy arrays
        
        # Ensure the queue is blocking for reliable calibration sample collection
        # This method assumes the queue passed is already configured appropriately (e.g. from get_output_queues)
        # but a check or specific queue creation for calibration might be more robust.

        start_time = time.monotonic()
        while len(gyro_readings_list) < num_samples:
            # Using blocking get() ensures we wait for data
            imu_data_packet = imu_queue.get() # This is dai.IMUData
            
            if imu_data_packet:
                imu_packets: List[dai.IMUPacket] = imu_data_packet.packets
                for packet in imu_packets:
                    # Use GYROSCOPE_CALIBRATED if that's what's enabled, or GYROSCOPE_RAW
                    gyro_report: dai.IMUReportGyroscope = packet.gyroscope
                    gyro_readings_list.append(np.array([gyro_report.x, gyro_report.y, gyro_report.z]))
                    if len(gyro_readings_list) >= num_samples:
                        break # Break inner loop
            if len(gyro_readings_list) >= num_samples:
                break # Break outer loop
            
            # Timeout safeguard if blocking get somehow fails or if non-blocking queue used
            if time.monotonic() - start_time > (num_samples * (1/self.imu_node.getReportRate() if self.imu_node else 100) + 5): # 5s grace
                print("Gyroscope calibration timeout. Not enough samples collected.")
                break
            if not imu_data_packet: # Should not happen with blocking get unless timeout on queue itself
                 time.sleep(wait_time_ms / 1000.0)


        if len(gyro_readings_list) >= num_samples: # Check if enough samples were collected
            self.gyro_bias = np.mean(np.array(gyro_readings_list), axis=0)
            self.gyro_calibrated = True
            print(f"Gyroscope calibrated. Bias (rad/s): {self.gyro_bias}")
        elif len(gyro_readings_list) > 0 : # Some samples, but not enough
             print(f"Gyroscope calibration incomplete: Collected {len(gyro_readings_list)}/{num_samples} samples. Using these for bias estimate.")
             self.gyro_bias = np.mean(np.array(gyro_readings_list), axis=0)
             self.gyro_calibrated = True # Still mark as calibrated, but with a warning
        else:
            print("Gyroscope calibration failed: No readings obtained. Gyro bias will remain zero.")
            self.gyro_calibrated = False # Explicitly set to false
            self.gyro_bias = np.zeros(3) # Ensure bias is zero


    def reset_orientation(self) -> None:
        """
        Resets the current estimated orientation to zero (roll, pitch, yaw = 0,0,0).
        Also resets the base timestamp for calculations.
        """
        self.reference_orientation_rad = np.array([0.0, 0.0, 0.0])
        self.base_ts = None # Reset base timestamp to re-anchor time calculations
        print("IMU orientation reset.")


    def process_imu_packet(self,
                           imu_packet: dai.IMUPacket,
                           last_gyro_ts_device: Optional[np.datetime64] = None
                           ) -> Optional[Dict[str, any]]:
        """
        Processes a single IMU packet to extract sensor data and update orientation.

        This method extracts accelerometer and gyroscope readings, applies gyroscope bias
        correction if calibrated, and integrates gyroscope data to update the
        `reference_orientation_rad`. The orientation is a simple Euler angle integration
        and will drift over time.

        Args:
            imu_packet (dai.IMUPacket): A single IMU packet from the DepthAI device.
            last_gyro_ts_device (Optional[np.datetime64]): Timestamp of the previous
                gyroscope reading from the device. Used to calculate the time delta (`dt`)
                for integration. If None, `dt` will be zero for the first packet.

        Returns:
            Optional[Dict[str, any]]: A dictionary containing processed IMU data:
                - "accel_ts_ms" (float): Accelerometer timestamp in milliseconds relative to base.
                - "accel_m_s2" (np.ndarray): Accelerometer readings (X,Y,Z) in m/s^2.
                - "gyro_ts_ms" (float): Gyroscope timestamp in milliseconds relative to base.
                - "gyro_rad_s" (np.ndarray): Gyroscope readings (X,Y,Z) in rad/s, bias corrected.
                - "orientation_deg" (np.ndarray): Estimated orientation (roll, pitch, yaw) in degrees.
                - "raw_gyro_rad_s" (np.ndarray): Raw gyroscope readings (X,Y,Z) in rad/s, uncorrected.
                - "new_last_gyro_ts_device" (np.datetime64): Current gyroscope timestamp, to be
                  passed as `last_gyro_ts_device` in the next call.
            Returns `None` if the input `imu_packet` is None or essential data is missing.
        """
        if not imu_packet:
            return None

        accel_report: dai.IMUReportAccelerometer = imu_packet.acceleroMeter
        gyro_report: dai.IMUReportGyroscope = imu_packet.gyroscope

        # Timestamps from device are typically timedelta from device boot/pipeline start
        accel_ts_device: np.datetime64 = accel_report.getTimestampDevice()
        gyro_ts_device: np.datetime64 = gyro_report.getTimestampDevice()


        if self.base_ts is None: # Initialize base timestamp with the first packet's timestamp
            self.base_ts = min(accel_ts_device, gyro_ts_device)

        # Calculate timestamps in milliseconds relative to the base_ts
        accel_ts_ms = self.time_delta_to_millis(accel_ts_device - self.base_ts)
        gyro_ts_ms = self.time_delta_to_millis(gyro_ts_device - self.base_ts)

        accel_vals_ms2 = np.array([accel_report.x, accel_report.y, accel_report.z])
        gyro_vals_rad_s_raw = np.array([gyro_report.x, gyro_report.y, gyro_report.z])

        if self.gyro_calibrated:
            gyro_vals_rad_s_corrected = gyro_vals_rad_s_raw - self.gyro_bias
        else:
            gyro_vals_rad_s_corrected = gyro_vals_rad_s_raw


        delta_t_s = 0.0
        if last_gyro_ts_device is not None:
            time_diff_ns = (gyro_ts_device - last_gyro_ts_device).astype('timedelta64[ns]').item()
            delta_t_s = time_diff_ns / 1.0e9 # Convert nanoseconds to seconds

        # Only integrate if delta_t_s is positive and reasonable
        if delta_t_s > 0 and delta_t_s < 1.0: # Avoid huge rotation if timestamps are weird
            self.reference_orientation_rad += gyro_vals_rad_s_corrected * delta_t_s
        elif delta_t_s < 0:
            # This can happen if packets are processed out of order or timestamps reset
            print(f"Warning: Negative delta_t_s ({delta_t_s:.4f}s) in IMU processing. Resetting last_gyro_ts_device.")
            # Consider not updating orientation or resetting last_gyro_ts_device to current to resync
            # For now, we just don't integrate.

        orientation_deg = np.degrees(self.reference_orientation_rad)

        return {
            "accel_ts_ms": accel_ts_ms,
            "accel_m_s2": accel_vals_ms2,
            "gyro_ts_ms": gyro_ts_ms,
            "gyro_rad_s": gyro_vals_rad_s_corrected,
            "orientation_deg": orientation_deg,
            "raw_gyro_rad_s": gyro_vals_rad_s_raw,
            "new_last_gyro_ts_device": gyro_ts_device # Pass this to next call
        }


if __name__ == '__main__':
    # This section is for conceptual demonstration and requires a running DepthAI pipeline.
    print("IMUHandler conceptual example:")

    # 1. Setup (Conceptual - requires a real pipeline and device)
    # try:
    #     mock_pipeline = dai.Pipeline()
    #     imu_handler = ImuHandler(mock_pipeline) # accel_hz and gyro_hz can be specified
    #     print("  IMUHandler initialized. IMU node configured in mock pipeline.")
    #     print(f"  IMU node settings: Accel Hz={imu_handler.imu_node.getReportRate()}, Gyro Hz (requested for type)={imu_handler.imu_node.getReportRate()}") # Note: getReportRate might give combined rate or for first sensor. Check docs.
    # except Exception as e:
    #     print(f"  Error during conceptual setup: {e}")
    #     imu_handler = None


    # 2. Getting Queues and Processing (Conceptual)
    # if imu_handler:
    #     print("\n  --- Conceptual Device Interaction ---")
    #     # `with dai.Device(mock_pipeline) as device:` would be here
    #     # For this example, we'll mock the device and queue interaction.
    #
    #     # Mocking a device and queue
    #     class MockDevice:
    #         def getOutputQueue(self, name, maxSize, blocking):
    #             print(f"    MockDevice: getOutputQueue called for '{name}'")
    #             return MockImuQueue() # Return a mock queue
    #
    #     class MockImuQueue:
    #         def get(self): # Simulate blocking get()
    #             print("      MockImuQueue: get() called, simulating data packet.")
    #             # Simulate receiving a packet after a short delay
    #             time.sleep(0.02) # ~50Hz
    #             return self._create_mock_imu_data()
    #
    #         def _create_mock_imu_data(self):
    #             class MockTimestamp:
    #                 def __init__(self, offset_s):
    #                     # Create a datetime64 like timestamp
    #                     self.ts = np.datetime64(int(time.time() * 1e9) + int(offset_s * 1e9), 'ns')
    #                 def getTimestampDevice(self): return self.ts
    #
    #             class MockAccel:
    #                 def __init__(self): self.x, self.y, self.z = 0.1, 0.1, 9.8
    #                 def getTimestampDevice(self): return MockTimestamp(0).ts # Offset for accel
    #
    #             class MockGyro:
    #                 def __init__(self): self.x, self.y, self.z = np.random.randn()*0.01, np.random.randn()*0.01, np.random.randn()*0.01 # Small random noise
    #                 def getTimestampDevice(self): return MockTimestamp(0.001).ts # Slightly offset for gyro
    #
    #             class MockImuPacket:
    #                 def __init__(self):
    #                     self.acceleroMeter = MockAccel()
    #                     self.gyroscope = MockGyro()
    #
    #             class MockImuData:
    #                 def __init__(self):
    #                     self.packets = [MockImuPacket()]
    #             return MockImuData()
    #
    #     mock_device_instance = MockDevice()
    #     imu_queues = imu_handler.get_output_queues(mock_device_instance)
    #     imu_q = imu_queues.get("imu")
    #
    #     if imu_q:
    #         print("    IMU queue obtained (mocked).")
    #
    #         # 3. Calibration
    #         print("    Simulating gyroscope calibration (with mocked data)...")
    #         imu_handler.calibrate_gyroscope(imu_q, num_samples=50) # Needs a real queue for actual calibration
    #         print(f"    Gyro calibrated: {imu_handler.gyro_calibrated}, Bias: {imu_handler.gyro_bias}")
    #         imu_handler.reset_orientation()
    #
    #         # 4. Processing Data Loop (simulated)
    #         print("    Simulating processing of IMU packets...")
    #         last_ts_device = None
    #         for i in range(5): # Simulate 5 packets
    #             imu_data = imu_q.get() # Get a packet from the mock queue
    #             if imu_data and imu_data.packets:
    #                 processed_data = imu_handler.process_imu_packet(imu_data.packets[0], last_gyro_ts_device=last_ts_device)
    #                 if processed_data:
    #                     print(f"      Packet {i+1}: Orient(deg): Roll={processed_data['orientation_deg'][0]:.2f}, Pitch={processed_data['orientation_deg'][1]:.2f}, Yaw={processed_data['orientation_deg'][2]:.2f}")
    #                     last_ts_device = processed_data['new_last_gyro_ts_device']
    #                 else:
    #                     print("      Mock processing failed for a packet.")
    #             time.sleep(0.01) # Short delay between processing
    #     else:
    #         print("    Mock IMU queue not obtained.")

    print("\nIMUHandler conceptual example complete.")
    print("To run this, a DepthAI device with an IMU is required, and the example needs to be part of a DepthAI pipeline execution.")
    print("Key functionalities demonstrated (conceptually): IMU node setup, calibration, data processing, and basic orientation tracking.")
    print("Note: Orientation tracking via direct integration will drift. For robust orientation, sensor fusion algorithms are necessary.")

```
