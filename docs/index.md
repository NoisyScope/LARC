# Eyes On Me - Documentation

Welcome to the documentation for the "Eyes On Me" project. This project is a DepthAI-based perception system designed for real-time object detection, spatial localization, and orientation tracking.

This documentation provides an overview of the system architecture, individual module functionalities, and the API reference for developers.

## Project Modules and Scripts

Below is a list of the core modules and scripts that make up the "Eyes On Me" project. Each link will take you to the detailed documentation for that component, extracted from its source code.

*   [**Main Application (`main.py`)**](./main.md)
    *   The primary script that initializes and runs the integrated perception pipeline.
*   [**Camera Manager (`src/camera_manager.py`)**](./camera_manager.md)
    *   Manages DepthAI camera setup (RGB, stereo), configuration, and frame acquisition.
*   [**Model Loader (`src/model_loader.py`)**](./model_loader.md)
    *   Handles loading and configuration of machine learning models for both DepthAI on-device inference and host-side TFLite inference.
*   [**Spatial Processor (`src/spatial_processing.py`)**](./spatial_processing.md)
    *   Processes depth frames and extracts 3D spatial coordinates (X, Y, Z) of detected objects.
*   [**IMU Handler (`src/imu_handler.py`)**](./imu_handler.md)
    *   Manages the IMU sensor, including setup, data fetching, calibration, and basic orientation calculation.
*   [**Utilities (`src/utils.py`)**](./utils.md)
    *   Provides helper functions and classes for tasks like FPS calculation, visualization, time conversion, and serial communication.

## Navigation

Use the links above to navigate to the specific module documentation. Each module's page will detail its classes, functions, methods, and their respective APIs including arguments, return values, and attributes.
```
