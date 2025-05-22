"""
Utility functions and classes for the DepthAI project.

This module provides helper functionalities such as FPS calculation,
visualization of detections and IMU data on frames, time conversion,
and serial communication. It also includes a specific visualization
function for TFLite detection results from the `tflite_support` library.
"""
import cv2
import numpy as np
import time
import serial # For serial communication, if used
from tflite_support.task import processor # For type hinting visualize_tflite_detections

# --- FPS Calculation ---
class FPSHandler:
    """
    Handles Frames Per Second (FPS) calculation.

    This class provides a simple way to track and calculate the FPS
    of a processing loop, typically a video stream.

    Attributes:
        start_time (float): The timestamp when the current counting period started.
        counter (int): The number of frames processed in the current period.
        fps (float): The most recently calculated FPS value.
    """
    def __init__(self):
        """Initializes the FPSHandler."""
        self.start_time = time.monotonic()
        self.counter = 0
        self.fps = 0.0

    def tick(self) -> float:
        """
        Increments the frame counter and recalculates FPS if a second has passed.

        This method should be called once for every frame processed.

        Returns:
            float: The current FPS value.
        """
        self.counter += 1
        current_time = time.monotonic()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.counter / elapsed_time
            self.counter = 0
            self.start_time = current_time
        return self.fps

    def get_fps(self) -> float:
        """
        Returns the current calculated FPS.

        Returns:
            float: The most recent FPS value.
        """
        return self.fps

# --- Visualization ---
def draw_detections(
    frame: np.ndarray,
    detections: list,
    spatial_processor=None,
    label_map: dict = None,
    display_confidence: bool = True
) -> None:
    """
    Draws bounding boxes, labels, confidence, and spatial coordinates for detections on a frame.

    This function iterates through a list of detection objects and renders their
    information onto the provided image frame. It supports both 2D bounding boxes
    and spatial coordinates if a `SpatialProcessor` is provided.

    Args:
        frame (np.ndarray): The image frame (OpenCV BGR format) on which to draw.
        detections (list): A list of detection objects. Each object should conform to
            the structure expected by this function (see below).
            For DepthAI YoloSpatialDetectionNetwork, detections usually have:
            - `label` (int or str): Class label index.
            - `confidence` (float): Detection confidence score.
            - `spatialCoordinates` (object): Contains `x`, `y`, `z` in millimeters.
            - `xmin`, `ymin`, `xmax`, `ymax` (float): Normalized bounding box coordinates (0-1).
            For other 2D detectors, ensure `xmin, ymin, xmax, ymax` are present.
        spatial_processor (SpatialProcessor, optional): An instance of `SpatialProcessor`.
            If provided, its `get_spatial_coordinates` method is used to obtain
            denormalized bounding box coordinates and spatial X,Y,Z values.
        label_map (dict, optional): A dictionary mapping label indices (int) to
            human-readable string names. E.g., `{0: "person", 1: "car"}`.
        display_confidence (bool): If True, the confidence score of the detection
            is displayed. Defaults to True.

    Example:
        >>> class MockSpatialProcessor:
        ...     def get_spatial_coordinates(self, det, frame_width, frame_height):
        ...         return (det.spatialCoordinates.x, det.spatialCoordinates.y, det.spatialCoordinates.z,
        ...                 int(det.xmin * frame_width), int(det.ymin * frame_height),
        ...                 int(det.xmax * frame_width), int(det.ymax * frame_height))
        >>> class MockDetection:
        ...     def __init__(self, label, conf, xmin, ymin, xmax, ymax, sx, sy, sz):
        ...         self.label = label; self.confidence = conf
        ...         self.xmin=xmin; self.ymin=ymin; self.xmax=xmax; self.ymax=ymax
        ...         class SC: pass
        ...         self.spatialCoordinates = SC(); self.spatialCoordinates.x=sx
        ...         self.spatialCoordinates.y=sy; self.spatialCoordinates.z=sz
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> detections = [MockDetection(0, 0.9, 0.1, 0.1, 0.3, 0.3, 100, 50, 1500)]
        >>> sp = MockSpatialProcessor()
        >>> lm = {0: "object"}
        >>> draw_detections(frame, detections, spatial_processor=sp, label_map=lm)
        >>> # Frame would now have the detection drawn on it.
    """
    height, width = frame.shape[:2]
    default_color = (255, 255, 255) # White for text
    box_color = (0, 255, 0) # Green for boxes

    for det in detections:
        label_text = ""
        if hasattr(det, 'label'):
            label_idx = int(det.label)
            if label_map and label_idx in label_map:
                label_text = label_map[label_idx]
            else:
                label_text = str(label_idx)

        # Initialize coordinates
        x, y, z = None, None, None
        xmin, ymin, xmax, ymax = 0, 0, 0, 0 # Default to prevent error if logic fails

        try:
            if spatial_processor and hasattr(det, 'spatialCoordinates'):
                # Use spatial_processor for coordinates (e.g. YoloSpatialDetectionNetwork)
                coords_result = spatial_processor.get_spatial_coordinates(
                    det, frame_width=width, frame_height=height
                )
                x, y, z, xmin, ymin, xmax, ymax = coords_result
            elif hasattr(det, 'xmin') and hasattr(det, 'xmax') and hasattr(det, 'ymin') and hasattr(det, 'ymax'):
                # Standard 2D detection with normalized coordinates
                xmin = int(det.xmin * width)
                xmax = int(det.xmax * width)
                ymin = int(det.ymin * height)
                ymax = int(det.ymax * height)
            else:
                # If detection format is unknown, skip drawing this detection
                # print(f"Warning: Unknown detection format: {vars(det)}")
                continue
        except Exception as e:
            # print(f"Error processing detection for drawing: {e}, Det: {vars(det)}")
            continue


        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)

        # Prepare text lines
        text_lines = []
        if label_text:
            text_lines.append(f"L: {label_text}")
        if hasattr(det, 'confidence') and display_confidence:
            text_lines.append(f"C: {det.confidence:.2f}")

        if x is not None and y is not None and z is not None: # Spatial coordinates were available
            text_lines.append(f"X: {int(x)}mm")
            text_lines.append(f"Y: {int(y)}mm")
            text_lines.append(f"Z: {int(z)}mm")

        # Display text
        y_offset = ymin + 15 # Start text slightly inside the top of the box
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (xmin + 5, y_offset + i * 15),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, default_color, 1)

def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Draws the Frames Per Second (FPS) count on the frame.

    Args:
        frame (np.ndarray): The image frame (OpenCV BGR format) on which to draw.
        fps (float): The FPS value to display.
    """
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_imu_data(frame: np.ndarray, imu_data: dict) -> None:
    """
    Draws IMU data (e.g., orientation) on the frame.

    Args:
        frame (np.ndarray): The image frame (OpenCV BGR format) on which to draw.
        imu_data (dict): A dictionary containing IMU data, typically from `ImuHandler`.
            Expected to have a key "orientation_deg" which is a list or numpy array
            of [roll, pitch, yaw] in degrees.
    """
    if not imu_data or "orientation_deg" not in imu_data:
        return

    orientation = imu_data["orientation_deg"]
    # Ensure orientation is iterable and has at least 3 elements
    try:
        roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
    except (IndexError, TypeError):
        # print("Warning: IMU data 'orientation_deg' has unexpected format.")
        return


    text_lines = [
        f"Roll: {roll:.2f} deg",
        f"Pitch: {pitch:.2f} deg",
        f"Yaw: {yaw:.2f} deg"
    ]

    # Display text (e.g., top-right corner)
    x_offset = frame.shape[1] - 200 # Adjust as needed
    y_offset = 20
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (x_offset, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)


# --- Time Conversion ---
def time_delta_to_millis(delta: 'time.timedelta') -> float:
    """
    Converts a `datetime.timedelta` object to milliseconds.

    Args:
        delta (datetime.timedelta): The timedelta object to convert.

    Returns:
        float: The equivalent duration in milliseconds.
    """
    return delta.total_seconds() * 1000.0


# --- Serial Communication ---
class SerialCommunicator:
    """
    Basic serial communication handler.

    Manages opening, closing, writing, and reading from a serial port.
    This class is a simple wrapper around `pySerial` for common operations.

    Attributes:
        port (str): The serial port identifier (e.g., '/dev/ttyUSB0', 'COM3').
        baudrate (int): The baud rate for serial communication.
        timeout (float): Read timeout in seconds.
        ser (serial.Serial): The pySerial `Serial` object instance.
        is_open (bool): True if the serial port is currently open, False otherwise.
    """
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600, timeout: float = 1.0):
        """
        Initializes the SerialCommunicator.

        Args:
            port (str): The serial port to connect to (e.g., '/dev/ttyUSB0', 'COM3').
            baudrate (int): The communication speed (e.g., 9600, 115200).
            timeout (float): Read timeout value in seconds.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_open = False

    def connect(self) -> bool:
        """
        Attempts to connect to the serial port.

        Returns:
            bool: True if connection was successful, False otherwise.

        Raises:
            serial.SerialException: If there is an error opening the serial port
                (though this method catches it and prints an error message).
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.is_open = True
            print(f"Successfully connected to serial port {self.port} at {self.baudrate} baud.")
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            self.ser = None # Ensure ser is None if connection failed
            self.is_open = False
            return False

    def close(self) -> None:
        """Closes the serial port if it's open."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_open = False
            print(f"Serial port {self.port} closed.")

    def send_data(self, data_string: str) -> bool:
        """
        Sends a string over the serial port.

        Appends a newline character `\\n` if it's not already present, as many
        serial devices expect newline-terminated commands.

        Args:
            data_string (str): The string to send.

        Returns:
            bool: True if data was sent successfully, False otherwise.
        """
        if not self.is_open or not self.ser:
            print("Serial port not open. Cannot send data.")
            return False
        try:
            if not data_string.endswith('\n'):
                data_string += '\n'
            self.ser.write(data_string.encode('utf-8'))
            # print(f"Sent: {data_string.strip()}") # Optional: for debugging
            return True
        except serial.SerialTimeoutException:
            print(f"Serial write timeout on port {self.port}.")
            return False
        except Exception as e: # Catch other potential serial errors
            print(f"Error sending data via serial port {self.port}: {e}")
            return False

    def read_line(self) -> str | None:
        """
        Reads a line from the serial port.

        A "line" is considered to be bytes up to a newline character `\\n`.
        The newline character is stripped from the returned string. This is a
        blocking call if `timeout` was set to None during init, otherwise it
        respects the timeout.

        Returns:
            str | None: The line read (UTF-8 decoded and stripped of newline),
            or None if a timeout occurred or no data was available, or an error occurred.
        """
        if not self.is_open or not self.ser:
            # print("Serial port not open. Cannot read data.")
            return None
        try:
            if self.ser.in_waiting > 0: # Check if there's data to be read
                line_bytes = self.ser.readline()
                return line_bytes.decode('utf-8').strip()
            return None # No data available within the timeout (if any)
        except Exception as e:
            print(f"Error reading data from serial port {self.port}: {e}")
            return None


if __name__ == '__main__':
    print("Utils conceptual examples:")

    # FPS Handler Example
    fps_handler = FPSHandler()
    print("  FPS Handler initialized.")
    for i in range(60): # Simulate 60 frames
        time.sleep(1/60.0) # Simulate ~60 FPS
        fps_handler.tick()
    print(f"  Calculated FPS after simulation: {fps_handler.get_fps():.2f}")

    # Visualization Example (Conceptual - requires a frame and detections)
    print("  Visualization functions: draw_detections, draw_fps, draw_imu_data.")
    # dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # class MockDet: pass
    # det1 = MockDet(); det1.label=0; det1.confidence=0.9; det1.xmin=0.1; det1.ymin=0.1; det1.xmax=0.3; det1.ymax=0.3
    # # Need a mock spatial_processor for this example to run as is for draw_detections
    # # draw_detections(dummy_frame, [det1], label_map={0: "object"})
    # draw_fps(dummy_frame, 30.0)
    # draw_imu_data(dummy_frame, {"orientation_deg": np.array([10.0, 20.0, 30.0])})
    # cv2.imshow("Utils Test", dummy_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Time Conversion Example
    from datetime import timedelta # Import timedelta for example
    td = timedelta(seconds=1.234)
    print(f"  Time delta {td} to milliseconds: {time_delta_to_millis(td)}")

    # Serial Communicator Example (Conceptual - requires a serial port or mock)
    print("  SerialCommunicator class defined.")
    # Use with a mock or a real serial loopback connection for testing.
    # ser_comm = SerialCommunicator(port='/dev/ttyS10') # Example, use a valid port if testing
    # if ser_comm.connect():
    #     ser_comm.send_data("Hello from utils_test")
    #     response = ser_comm.read_line()
    #     if response:
    #         print(f"  Serial read: {response}")
    #     ser_comm.close()
    # else:
    #     print("  Serial connection failed (as expected if port is dummy).")

    print("\nUtils conceptual examples complete.")


# --- Visualization for TFLite Detections (from tflite_support) ---
def visualize_tflite_detections(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  """Draws bounding boxes from TFLite Support `DetectionResult` on the input image.

  This function is specifically designed to work with the `DetectionResult` object
  provided by the TensorFlow Lite Task Library's `ObjectDetector`.

  Args:
    image (np.ndarray): The input RGB image on which to draw.
    detection_result (tflite_support.task.processor.DetectionResult):
        The object detection result from a TFLite Task Library detector.
        It contains a list of `Detection` objects.

  Returns:
    np.ndarray: The input image with bounding boxes and labels drawn onto it.
  """
  _MARGIN = 10  # pixels
  _ROW_SIZE = 10  # pixels
  _FONT_SIZE = 1
  _FONT_THICKNESS = 1
  _TEXT_COLOR = (0, 0, 255)  # BGR color for red

  if not detection_result or not detection_result.detections:
      return image # Return original image if no detections

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0] # Assuming one category per detection
    category_name = category.category_name if category.category_name else "N/A"
    probability = round(category.score, 2)
    result_text = f"{category_name} ({probability})"
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image

```
