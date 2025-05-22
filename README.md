# Eyes On Me - Object Detection and Spatial Localization Project

## Overview

This project implements a system for real-time object detection, spatial localization (3D coordinates), and IMU (Inertial Measurement Unit) data processing using Luxonis OAK devices. It leverages DepthAI API, custom machine learning models, and provides a modular structure for easy extension and experimentation.

The system can:
- Initialize and configure OAK cameras (RGB, stereo).
- Load and run object detection models (YOLO `.blob` format, TFLite for host processing).
- Calculate spatial coordinates (X, Y, Z) of detected objects.
- Integrate IMU data for orientation and rotation tracking.
- Visualize detections and spatial information on frames.
- Send data over serial communication (optional).

## Project Structure

```
.
├── docs/                   # Project documentation files in Markdown format
│   ├── index.md            # Main entry point for documentation
│   ├── main.md
│   ├── camera_manager.md
│   ├── model_loader.md
│   ├── spatial_processing.md
│   ├── imu_handler.md
│   └── utils.md
├── examples/
│   └── MLtest.py           # Standalone TFLite object detection example using webcam
├── models/                 # Directory for ML model files (e.g., .blob, .tflite)
│   └── (User needs to place model files here, e.g., best_yolov7_v1_openvino_2022.1_6shave.blob)
├── src/                    # Source code for the main application
│   ├── __init__.py
│   ├── camera_manager.py   # Manages OAK camera setup and frame acquisition
│   ├── model_loader.py     # Loads and runs ML models (DepthAI NN, TFLite)
│   ├── spatial_processing.py # Handles depth data and 3D coordinate calculation
│   ├── imu_handler.py      # Manages IMU sensor interaction and data processing
│   └── utils.py            # Utility functions (visualization, FPS, serial comms)
├── main.py                 # Main application script to run the integrated pipeline
└── README.md               # This file
```

## Setup and Installation

**1. Hardware Requirements:**
   - Luxonis OAK camera (e.g., OAK-D, OAK-D Lite)
   - Computer with Python environment (Linux recommended for full DepthAI compatibility)
   - (Optional) Coral EdgeTPU for `examples/MLtest.py` if using EdgeTPU models.

**2. Software Dependencies:**
   - Python 3.7+
   - DepthAI library
   - OpenCV
   - NumPy
   - (Optional, for TFLite example) TensorFlow Lite Runtime (`tflite_runtime` or `tensorflow`)
   - (Optional, for TFLite example) TFLite Support library (`tflite-support`)

   Install dependencies using pip:
   ```bash
   pip install depthai numpy opencv-python
   # For examples/MLtest.py and TFLite model loading in general:
   pip install tflite-support # Or tflite_runtime, depending on your needs
   ```
   It's recommended to use a virtual environment.

**3. Model Files:**
   - This project uses pre-trained models for object detection. You need to place your model files (e.g., `best_yolov7_v1_openvino_2022.1_6shave.blob`, `efficientdet_lite0.tflite`) into the `models/` directory.
   - The default YOLO model path in `main.py` is configured for `models/best_yolov7_v1_openvino_2022.1_6shave.blob`. Please update this path in `main.py` if your model file has a different name or you wish to use a different model.

## How to Run

**1. Main Application (OAK camera):**
   - Ensure your OAK camera is connected.
   - Place the required `.blob` model file in the `models/` directory.
   - Run the main script:
     ```bash
     python main.py
     ```
   - Configuration options (like model path, confidence thresholds, enabling/disabling IMU) can be adjusted within `main.py`.

**2. TFLite Example (Webcam):**
   - This example uses a standard webcam and a `.tflite` model.
   - Place your `.tflite` model (e.g., `efficientdet_lite0.tflite`) in the `models/` directory.
   - Run the example script:
     ```bash
     python examples/MLtest.py --model models/your_model_name.tflite
     ```
   - You might need to adjust the `--cameraId` if you have multiple webcams.

## Documentation

Project documentation, including an API reference, is available in Markdown format in the `docs/` directory.

- **[Start Reading the Documentation](./docs/index.md)**

The documentation details the project architecture, individual module functionalities, and provides an API reference for developers, extracted from the source code docstrings.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details (if a `LICENSE` file is added, otherwise assume this or specify).
```
