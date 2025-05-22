"""
Main application script for the DepthAI-based perception system.

This script initializes and orchestrates various components including camera management,
model loading and inference, spatial processing for 3D localization, IMU handling for
orientation, and serial communication. It sets up a DepthAI pipeline, processes
the output streams, and visualizes the results.

The application is configurable through global variables at the beginning of the script,
allowing adjustments to model paths, camera settings, IMU usage, visualization options,
and serial port parameters.

Key functionalities:
- Initializes `CameraManager` for RGB and depth camera streams.
- Initializes `ModelLoader` for loading a neural network model (e.g., YOLO for object detection).
- Links camera outputs to the model inputs.
- Optionally initializes `ImuHandler` for IMU data processing and orientation.
- Initializes `SpatialProcessor` for depth frame processing and 3D coordinate extraction.
- Initializes `FPSHandler` for performance monitoring.
- Optionally initializes `SerialCommunicator` for sending data to an external device.
- Enters a main loop that:
    - Retrieves frames and data from the DepthAI device.
    - Processes RGB frames, depth frames, and model detections.
    - Processes IMU data if enabled.
    - Visualizes detections, IMU data, and FPS on the output frame.
    - Handles user input for quitting ('q') or resetting IMU ('r').
"""
import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, List # For type hinting

from src.camera_manager import CameraManager
from src.model_loader import ModelLoader
from src.spatial_processing import SpatialProcessor
from src.imu_handler import ImuHandler
from src.utils import FPSHandler, draw_detections, draw_fps, draw_imu_data, SerialCommunicator

# --- Configuration ---
# Model configuration (adjust paths and parameters as needed)
NN_MODEL_NAME: str = "yolo" # "yolo" or a generic blob name
# Path to the Yolo blob file (e.g., from imutest.py)
NN_BLOB_PATH_CONFIG: str = '/home/pi/Desktop/eyes_on_me/models/yolov7_v2/best_yolov7_v1_openvino_2022.1_6shave.blob'

# Check if the default path exists, if not, use a dummy path for placeholder
# This allows the script to be structured and potentially run even if the specific model blob is missing,
# which can be useful for initial setup or testing parts of the pipeline.
if not Path(NN_BLOB_PATH_CONFIG).exists():
    print(f"Warning: Model blob not found at {NN_BLOB_PATH_CONFIG}. "
          "Attempting to create a dummy file for pipeline construction.")
    dummy_blob_path = Path("dummy_model.blob")
    try:
        dummy_blob_path.touch() # Create an empty file
        NN_BLOB_PATH: str = str(dummy_blob_path)
        print(f"Using dummy model path: {NN_BLOB_PATH}")
        # Define minimal valid Yolo config for dummy blob, otherwise DepthAI might crash during pipeline compilation.
        YOLO_CONFIG: Dict[str, Any] = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "num_classes": 1, # Minimal number of classes
            "coordinate_size": 4,
            "anchors": [10,13, 16,30, 33,23], # Minimal valid set for one anchor mask
            "anchor_masks": {"side20": [0,1,2]}, # Minimal valid anchor mask
            "input_size": (320, 320), # A common small YOLO input size
            "depth_lower_thresh": 100, # mm
            "depth_upper_thresh": 5000, # mm
        }
    except OSError as e:
        print(f"Error: Could not create dummy model file at {dummy_blob_path}: {e}. Please check permissions.")
        print("Attempting to proceed without a valid model path. This will likely fail if a model is required.")
        NN_BLOB_PATH = "invalid_dummy_model.blob" # Path that won't exist
        YOLO_CONFIG = {} # Empty config as the model path is invalid

else:
    NN_BLOB_PATH = NN_BLOB_PATH_CONFIG
    YOLO_CONFIG = { # Based on imutest.py for the specific model, adjust as needed
        "confidence_threshold": 0.5,
        "iou_threshold": 0.5,
        "num_classes": 6, # Example: 6 classes for the yolov7 model
        "coordinate_size": 4, # Typically 4 for bounding box coordinates
        "anchors": [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326], # Anchors for YOLO
        "anchor_masks": { "side80": [0,1,2], "side40": [3,4,5], "side20": [6,7,8] }, # Anchor masks for YOLO
        "input_size": (640, 640), # (width, height) - from imutest.py camRgb.setPreviewSize
        "depth_lower_thresh": 100,  # Minimum depth for spatial calculations (mm)
        "depth_upper_thresh": 5000, # Maximum depth for spatial calculations (mm)
    }


# Camera configuration based on model input size
PREVIEW_WIDTH: int = YOLO_CONFIG.get("input_size", (640, 640))[0]
PREVIEW_HEIGHT: int = YOLO_CONFIG.get("input_size", (640, 640))[1]

# IMU configuration
ENABLE_IMU: bool = True
IMU_CALIBRATION_SAMPLES: int = 200 # Number of samples for gyroscope calibration

# Visualization
ENABLE_VISUALIZATION: bool = True
# Example label map, adjust based on your model's classes and training.
# The keys are class indices (integer), and values are human-readable names (string).
LABEL_MAP: Dict[int, str] = {
    0: "person", 1: "vehicle", 2: "object_A", 3: "object_B", 4: "box_green", 5: "box_red"
    # Add more labels as per your YOLO model's classes (up to num_classes-1)
}
if YOLO_CONFIG.get("num_classes") == 1 and Path(NN_BLOB_PATH).name == "dummy_model.blob":
    LABEL_MAP = {0: "dummy_obj"}


# Serial communication (optional)
ENABLE_SERIAL: bool = False # Set to True to enable sending data over serial
SERIAL_PORT: str = '/dev/ttyUSB0' # Adjust to your serial port (e.g., 'COM3' on Windows)
SERIAL_BAUDRATE: int = 9600


def main() -> None:
    """
    Main function to run the DepthAI perception application.

    Initializes all components, sets up the DepthAI pipeline, connects to the
    device, and enters the main processing loop. Handles data retrieval,
    processing, visualization, and user interaction.

    The function will attempt to gracefully handle errors during initialization
    and execution, such as device connection issues or missing model files.
    It ensures resources like serial ports and OpenCV windows are cleaned up
    on exit.
    """
    print("Starting DepthAI application...")

    # Initialize components
    # CameraManager sets up the RGB and mono cameras, and the stereo depth node.
    cam_manager = CameraManager(preview_size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
                                nn_input_size=(PREVIEW_WIDTH, PREVIEW_HEIGHT), # For YOLO, NN input is typically the preview size
                                model_name=NN_MODEL_NAME)

    # ModelLoader loads the specified neural network model into the pipeline.
    # It requires the pipeline from CameraManager to add its NN node.
    model_loader = ModelLoader(pipeline=cam_manager.pipeline,
                               model_name=NN_MODEL_NAME,
                               model_path=NN_BLOB_PATH,
                               config=YOLO_CONFIG if NN_MODEL_NAME == "yolo" else {})

    # Get the NN node from ModelLoader and link it with camera inputs in CameraManager.
    nn_node = model_loader.get_model_node()
    if nn_node:
        try:
            cam_manager.link_model_inputs(nn_node)
        except Exception as e:
            print(f"Error linking model inputs: {e}. Check model compatibility and CameraManager setup.")
            if Path(NN_BLOB_PATH).name == "dummy_model.blob":
                 Path(NN_BLOB_PATH).unlink(missing_ok=True) # Clean up dummy file
            return
    else:
        print("Error: Could not get model node from ModelLoader. This might be due to an invalid model path or configuration. Exiting.")
        if Path(NN_BLOB_PATH).name == "dummy_model.blob":
             Path(NN_BLOB_PATH).unlink(missing_ok=True) # Clean up dummy file
        return

    imu_handler: Optional[ImuHandler] = None
    if ENABLE_IMU:
        imu_handler = ImuHandler(cam_manager.pipeline)

    spatial_processor = SpatialProcessor()
    fps_handler = FPSHandler()

    serial_comm: Optional[SerialCommunicator] = None
    if ENABLE_SERIAL:
        serial_comm = SerialCommunicator(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE)
        if not serial_comm.connect():
            print(f"Warning: Failed to connect to serial port {SERIAL_PORT}. Continuing without serial communication.")
            # serial_comm = None # Effectively disables serial if connection fails

    # --- Pipeline Setup and Device Connection ---
    try:
        with dai.Device(cam_manager.pipeline) as device:
            print("DepthAI device connected successfully.")

            # Get output queues from the managers
            cam_queues = cam_manager.get_output_queues(device)
            rgb_queue = cam_queues.get("rgb") # For YOLO, this is from NN passthrough for sync
            depth_queue = cam_queues.get("depth")

            nn_queues = model_loader.get_output_queues(device)
            detections_queue = nn_queues.get("detections")

            imu_queue: Optional[dai.DataOutputQueue] = None
            if ENABLE_IMU and imu_handler:
                imu_queues_dict = imu_handler.get_output_queues(device)
                imu_queue = imu_queues_dict.get("imu")
                if imu_queue:
                    print("Attempting IMU gyroscope calibration. Keep the device still...")
                    imu_handler.calibrate_gyroscope(imu_queue, num_samples=IMU_CALIBRATION_SAMPLES)
                    imu_handler.reset_orientation() # Start with a fresh (0,0,0) orientation
                else:
                    print("Warning: IMU queue not available from ImuHandler. IMU features will be disabled.")

            if not rgb_queue or not depth_queue or not detections_queue:
                print("Error: One or more critical output queues (RGB, Depth, Detections) are not available. Exiting.")
                return

            last_imu_packet_ts: Optional[np.datetime64] = None # For calculating delta_t in IMU processing

            while True:
                fps_handler.tick()
                current_fps = fps_handler.get_fps()

                # Non-blocking attempts to get data from queues
                rgb_frame_data: Optional[dai.ImgFrame] = rgb_queue.tryGet()
                depth_frame_data: Optional[dai.ImgFrame] = depth_queue.tryGet()
                detections_data: Optional[dai.SpatialImgDetections] = detections_queue.tryGet() # Assuming YoloSpatialDetectionNetwork
                
                imu_data_packet_list: Optional[List[dai.IMUData]] = None
                if ENABLE_IMU and imu_queue and imu_handler: # Check imu_handler too
                    imu_data_packet_list = imu_queue.tryGetAll()


                # Process RGB frame for display
                display_frame: Optional[np.ndarray] = None
                if rgb_frame_data:
                    display_frame = rgb_frame_data.getCvFrame()
                else: # If no RGB frame yet, create a blank one for messages or visualization
                    display_frame = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
                    # Optional: display a message if no RGB input
                    # cv2.putText(display_frame, "No RGB input", (PREVIEW_WIDTH//3, PREVIEW_HEIGHT//2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


                # Process Depth frame for visualization
                processed_depth_frame_color: Optional[np.ndarray] = None
                # raw_depth_frame: Optional[np.ndarray] = None # If needed for other purposes
                if depth_frame_data:
                    raw_depth_frame_from_device = depth_frame_data.getFrame()
                    processed_depth_frame_color, _ = spatial_processor.process_depth_frame(raw_depth_frame_from_device)


                # Process Detections
                current_detections: List[Any] = [] # Using Any for detection type from queue
                if detections_data:
                    current_detections = detections_data.detections
                    # Draw detections on the display_frame
                    if display_frame is not None:
                        draw_detections(display_frame, current_detections, spatial_processor, LABEL_MAP)

                    # Send detection data over serial (example format)
                    if ENABLE_SERIAL and serial_comm and serial_comm.is_open and current_detections:
                        data_strings_to_send: List[str] = []
                        for det in current_detections:
                            # Ensure spatialCoordinates attribute exists (it does for YoloSpatialDetectionNetwork)
                            if hasattr(det, 'spatialCoordinates'):
                                sp_coords = det.spatialCoordinates
                                # Format: label_index,confidence,X_mm,Y_mm,Z_mm
                                data_strings_to_send.append(
                                    f"{det.label},{det.confidence:.2f},"
                                    f"{sp_coords.x:.0f},{sp_coords.y:.0f},{sp_coords.z:.0f}"
                                )
                        if data_strings_to_send:
                            serial_comm.send_data(";".join(data_strings_to_send))


                # Process IMU data
                processed_imu_info: Optional[Dict[str, Any]] = None
                if ENABLE_IMU and imu_handler and imu_data_packet_list:
                    for imu_packet_group in imu_data_packet_list: # List of dai.IMUData
                        imu_packets_from_group: List[dai.IMUPacket] = imu_packet_group.packets
                        for imu_packet in imu_packets_from_group:
                            # Process each packet, updating orientation internally in imu_handler
                            processed_imu_info = imu_handler.process_imu_packet(imu_packet, last_imu_packet_ts)
                            if processed_imu_info:
                                last_imu_packet_ts = processed_imu_info["new_last_gyro_ts_device"]
                    # Draw the latest processed IMU data on the display_frame
                    if processed_imu_info and display_frame is not None:
                         draw_imu_data(display_frame, processed_imu_info)


                # Draw FPS on the display_frame
                if display_frame is not None:
                    draw_fps(display_frame, current_fps)

                # Show frames if visualization is enabled
                if ENABLE_VISUALIZATION and display_frame is not None:
                    cv2.imshow("RGB Feed", display_frame)
                    if processed_depth_frame_color is not None:
                        cv2.imshow("Depth Feed", processed_depth_frame_color)

                # Handle user input
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("User pressed 'q'. Exiting...")
                    break
                elif key == ord('r') and ENABLE_IMU and imu_handler: # Reset IMU orientation
                    imu_handler.reset_orientation()
                    last_imu_packet_ts = None # Reset this to avoid large delta_t on next packet
                    print("IMU orientation reset by user.")


    except RuntimeError as e:
        print(f"RuntimeError encountered: {e}")
        print("This might be due to no OAK device being connected, a model path issue, or pipeline misconfiguration.")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        # Cleanup resources
        if ENABLE_SERIAL and serial_comm and serial_comm.is_open:
            serial_comm.close()
        cv2.destroyAllWindows()
        # Clean up dummy model file if it was created
        if Path(NN_BLOB_PATH).name == "dummy_model.blob" and Path(NN_BLOB_PATH).exists():
            Path(NN_BLOB_PATH).unlink()
            print("Dummy model file cleaned up.")
        print("Application finished.")


if __name__ == "__main__":
    main()
```
