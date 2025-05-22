# Module main

> Main application script for the DepthAI-based perception system.
>
> This script initializes and orchestrates various components including camera management,
> model loading and inference, spatial processing for 3D localization, IMU handling for
> orientation, and serial communication. It sets up a DepthAI pipeline, processes
> the output streams, and visualizes the results.
>
> The application is configurable through global variables at the beginning of the script,
> allowing adjustments to model paths, camera settings, IMU usage, visualization options,
> and serial port parameters.
>
> Key functionalities:
> - Initializes `CameraManager` for RGB and depth camera streams.
> - Initializes `ModelLoader` for loading a neural network model (e.g., YOLO for object detection).
> - Links camera outputs to the model inputs.
> - Optionally initializes `ImuHandler` for IMU data processing and orientation.
> - Initializes `SpatialProcessor` for depth frame processing and 3D coordinate extraction.
> - Initializes `FPSHandler` for performance monitoring.
> - Optionally initializes `SerialCommunicator` for sending data to an external device.
> - Enters a main loop that:
>     - Retrieves frames and data from the DepthAI device.
>     - Processes RGB frames, depth frames, and model detections.
>     - Processes IMU data if enabled.
>     - Visualizes detections, IMU data, and FPS on the output frame.
>     - Handles user input for quitting ('q') or resetting IMU ('r').

---
## Functions

### `main`
```python
def main() -> None:
```
Main function to run the DepthAI perception application.

Initializes all components, sets up the DepthAI pipeline, connects to the
device, and enters the main processing loop. Handles data retrieval,
processing, visualization, and user interaction.

The function will attempt to gracefully handle errors during initialization
and execution, such as device connection issues or missing model files.
It ensures resources like serial ports and OpenCV windows are cleaned up
on exit.

---
```
