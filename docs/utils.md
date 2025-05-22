# Module src.utils

> Utility functions and classes for the DepthAI project.
>
> This module provides helper functionalities such as FPS calculation,
> visualization of detections and IMU data on frames, time conversion,
> and serial communication. It also includes a specific visualization
> function for TFLite detection results from the `tflite_support` library.

---
## Class FPSHandler

Handles Frames Per Second (FPS) calculation.

This class provides a simple way to track and calculate the FPS
of a processing loop, typically a video stream.

#### Attributes
*   **start_time** (*float*): The timestamp when the current counting period started.
*   **counter** (*int*): The number of frames processed in the current period.
*   **fps** (*float*): The most recently calculated FPS value.

### Methods
*   `__init__`
*   `tick`
*   `get_fps`

---
### `__init__`
```python
def __init__(self):
```
Initializes the FPSHandler.

---
### `tick`
```python
def tick(self) -> float:
```
Increments the frame counter and recalculates FPS if a second has passed.

This method should be called once for every frame processed.

#### Returns
*   (*float*): The current FPS value.

---
### `get_fps`
```python
def get_fps(self) -> float:
```
Returns the current calculated FPS.

#### Returns
*   (*float*): The most recent FPS value.

---
## Functions

### `draw_detections`
```python
def draw_detections(frame: np.ndarray, detections: list, spatial_processor=None, label_map: dict = None, display_confidence: bool = True) -> None:
```
Draws bounding boxes, labels, confidence, and spatial coordinates for detections on a frame.

This function iterates through a list of detection objects and renders their
information onto the provided image frame. It supports both 2D bounding boxes
and spatial coordinates if a `SpatialProcessor` is provided.

#### Arguments
*   **frame** (*np.ndarray*): The image frame (OpenCV BGR format) on which to draw.
*   **detections** (*list*): A list of detection objects. Each object should conform to
    the structure expected by this function (see below).
    For DepthAI YoloSpatialDetectionNetwork, detections usually have:
    - `label` (int or str): Class label index.
    - `confidence` (float): Detection confidence score.
    - `spatialCoordinates` (object): Contains `x`, `y`, `z` in millimeters.
    - `xmin`, `ymin`, `xmax`, `ymax` (float): Normalized bounding box coordinates (0-1).
    For other 2D detectors, ensure `xmin, ymin, xmax, ymax` are present.
*   **spatial_processor** (*SpatialProcessor, optional*): An instance of `SpatialProcessor`.
    If provided, its `get_spatial_coordinates` method is used to obtain
    denormalized bounding box coordinates and spatial X,Y,Z values.
*   **label_map** (*dict, optional*): A dictionary mapping label indices (int) to
    human-readable string names. E.g., `{0: "person", 1: "car"}`.
*   **display_confidence** (*bool, optional*): If True, the confidence score of the detection
    is displayed. Defaults to True.

#### Examples
```python
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
```

---
### `draw_fps`
```python
def draw_fps(frame: np.ndarray, fps: float) -> None:
```
Draws the Frames Per Second (FPS) count on the frame.

#### Arguments
*   **frame** (*np.ndarray*): The image frame (OpenCV BGR format) on which to draw.
*   **fps** (*float*): The FPS value to display.

---
### `draw_imu_data`
```python
def draw_imu_data(frame: np.ndarray, imu_data: dict) -> None:
```
Draws IMU data (e.g., orientation) on the frame.

#### Arguments
*   **frame** (*np.ndarray*): The image frame (OpenCV BGR format) on which to draw.
*   **imu_data** (*dict*): A dictionary containing IMU data, typically from `ImuHandler`.
    Expected to have a key "orientation_deg" which is a list or numpy array
    of [roll, pitch, yaw] in degrees.

---
### `time_delta_to_millis`
```python
def time_delta_to_millis(delta: 'time.timedelta') -> float:
```
Converts a `datetime.timedelta` object to milliseconds.

#### Arguments
*   **delta** (*datetime.timedelta*): The timedelta object to convert.

#### Returns
*   (*float*): The equivalent duration in milliseconds.

---
## Class SerialCommunicator

Basic serial communication handler.

Manages opening, closing, writing, and reading from a serial port.
This class is a simple wrapper around `pySerial` for common operations.

#### Attributes
*   **port** (*str*): The serial port identifier (e.g., '/dev/ttyUSB0', 'COM3').
*   **baudrate** (*int*): The baud rate for serial communication.
*   **timeout** (*float*): Read timeout in seconds.
*   **ser** (*serial.Serial*): The pySerial `Serial` object instance.
*   **is_open** (*bool*): True if the serial port is currently open, False otherwise.

### Methods
*   `__init__`
*   `connect`
*   `close`
*   `send_data`
*   `read_line`

---
### `__init__`
```python
def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600, timeout: float = 1.0):
```
Initializes the SerialCommunicator.

#### Arguments
*   **port** (*str, optional*): The serial port to connect to (e.g., '/dev/ttyUSB0', 'COM3'). Defaults to '/dev/ttyUSB0'.
*   **baudrate** (*int, optional*): The communication speed (e.g., 9600, 115200). Defaults to 9600.
*   **timeout** (*float, optional*): Read timeout value in seconds. Defaults to 1.0.

---
### `connect`
```python
def connect(self) -> bool:
```
Attempts to connect to the serial port.

#### Returns
*   (*bool*): True if connection was successful, False otherwise.

#### Raises
*   **serial.SerialException**: If there is an error opening the serial port
    (though this method catches it and prints an error message).

---
### `close`
```python
def close(self) -> None:
```
Closes the serial port if it's open.

---
### `send_data`
```python
def send_data(self, data_string: str) -> bool:
```
Sends a string over the serial port.

Appends a newline character `\n` if it's not already present, as many
serial devices expect newline-terminated commands.

#### Arguments
*   **data_string** (*str*): The string to send.

#### Returns
*   (*bool*): True if data was sent successfully, False otherwise.

---
### `read_line`
```python
def read_line(self) -> str | None:
```
Reads a line from the serial port.

A "line" is considered to be bytes up to a newline character `\n`.
The newline character is stripped from the returned string. This is a
blocking call if `timeout` was set to None during init, otherwise it
respects the timeout.

#### Returns
*   (*str | None*): The line read (UTF-8 decoded and stripped of newline),
    or None if a timeout occurred or no data was available, or an error occurred.

---
### `visualize_tflite_detections`
```python
def visualize_tflite_detections(image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
```
Draws bounding boxes from TFLite Support `DetectionResult` on the input image.

This function is specifically designed to work with the `DetectionResult` object
provided by the TensorFlow Lite Task Library's `ObjectDetector`.

#### Arguments
*   **image** (*np.ndarray*): The input RGB image on which to draw.
*   **detection_result** (*tflite_support.task.processor.DetectionResult*):
    The object detection result from a TFLite Task Library detector.
    It contains a list of `Detection` objects.

#### Returns
*   (*np.ndarray*): The input image with bounding boxes and labels drawn onto it.

---
```
