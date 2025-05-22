"""
Machine Learning Model Loader for DepthAI and Host Processing.

This module provides the `ModelLoader` class, designed to handle the
loading and configuration of machine learning models for both DepthAI
on-device inference (using `.blob` files) and host-side inference
(using `.tflite` files with TensorFlow Lite Runtime/Support).

It supports different model architectures like YOLO for spatial detection
on DepthAI, generic neural network blobs for DepthAI, and TFLite models
for execution on the host CPU or EdgeTPU.
"""
import depthai as dai
from pathlib import Path
from tflite_support.task import core, processor, vision
import cv2 # For MLtest.py's TFLite part, if we adapt it.
import numpy as np # Added for dummy frame in TFLite example
from typing import Optional, Dict, Any, Tuple, Union # For type hinting

class ModelLoader:
    """
    Manages loading and running ML models on DepthAI or the host.

    This class encapsulates the logic for initializing different types of
    neural network models, whether they are DepthAI-specific blob files
    or TensorFlow Lite models for host-side execution. It configures
    the model based on provided parameters and prepares it for inference.

    Attributes:
        pipeline (Optional[dai.Pipeline]): The DepthAI pipeline object. Required for
            DepthAI blob models to add the NN node.
        model_name (str): The name or type of the model (e.g., "yolo", "efficientdet_tflite").
        model_path (Optional[Path]): The file path to the model (.blob or .tflite).
        config (Dict[str, Any]): Configuration dictionary for model-specific parameters.
        nn_node (Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]]):
            The DepthAI neural network node instance, if a DepthAI model is loaded.
        xout_nn (Optional[dai.node.XLinkOut]): The XLinkOut node for DepthAI model's
            detection results.
        tflite_detector (Optional[vision.ObjectDetector]): The TFLite ObjectDetector
            instance from `tflite_support`, if a TFLite model is loaded for host processing.
    """
    def __init__(self,
                 pipeline: Optional[dai.Pipeline] = None,
                 model_name: str = "unknown",
                 model_path: Optional[Union[str, Path]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ModelLoader.

        Based on the `model_name`, `model_path` suffix, and whether a `pipeline`
        is provided, this constructor will attempt to initialize either a DepthAI
        neural network node or a host-side TFLite detector.

        Args:
            pipeline (Optional[dai.Pipeline]): The DepthAI pipeline object to which
                a DepthAI NN node should be added. Required for `.blob` models.
                Defaults to None.
            model_name (str): Name/type of the model. Used to determine setup logic.
                Examples: "yolo", "mobilenet_ssd", "efficientdet_tflite".
                Defaults to "unknown".
            model_path (Optional[Union[str, Path]]): Path to the model file.
                Can be a `.blob` for DepthAI or `.tflite` for host processing.
                Defaults to None.
            config (Optional[Dict[str, Any]]): Model-specific configurations.
                See class documentation for examples for YOLO and TFLite.
                Defaults to an empty dictionary.

        Raises:
            FileNotFoundError: If `model_path` is provided but the file does not exist.

        Example Config for YOLO (DepthAI):
            {
                "confidence_threshold": 0.5, "iou_threshold": 0.5,
                "num_classes": 80, "coordinate_size": 4,
                "anchors": [...], "anchor_masks": {...},
                "input_size": (416, 416) # (width, height)
            }

        Example Config for TFLite (host):
            {
                "enable_edgetpu": False, "num_threads": 4,
                "score_threshold": 0.5, "max_results": 5,
                "input_size": (300, 300) # (width, height), optional
            }
        """
        self.pipeline = pipeline
        self.model_name = model_name.lower()
        self.model_path = Path(model_path) if model_path else None
        self.config = config if config else {}
        self.nn_node: Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]] = None
        self.xout_nn: Optional[dai.node.XLinkOut] = None
        self.tflite_detector: Optional[vision.ObjectDetector] = None

        if not self.model_path or not self.model_path.exists():
            if model_path: # Only raise error if a path was explicitly given and it's invalid
                 raise FileNotFoundError(f"Model file not found: {self.model_path}")
            # Allow initialization without a model_path if none was provided.
            # Useful if configuration is deferred or if class is used for other purposes.
            return


        if self.model_name == "yolo" and self.pipeline and self.model_path.suffix == '.blob':
            self._create_yolo_spatial_network()
        elif self.model_name.endswith("_tflite") and self.model_path.suffix == '.tflite':
            self._initialize_tflite_detector()
        elif self.model_path.suffix == '.blob' and self.pipeline: # Generic .blob model for DepthAI
            self._create_generic_nn_network()
        elif self.model_path.suffix == '.tflite': # If model_name wasn't specific but it's a tflite file
             print(f"Warning: .tflite file provided ({self.model_path}) but model_name is '{self.model_name}'. "
                   f"Assuming TFLite host model based on suffix.")
             self.model_name = self.model_path.stem + "_tflite" # Auto-correct model_name
             self._initialize_tflite_detector()
        else:
            err_msg = f"Model type for path '{self.model_path}' with name '{self.model_name}' is not properly handled."
            if self.model_path.suffix == '.blob' and not self.pipeline:
                err_msg += " A DepthAI pipeline instance is required for blob models."
            print(f"Warning: {err_msg}")


    def _create_yolo_spatial_network(self) -> None:
        """
        Creates and configures a YoloSpatialDetectionNetwork node for DepthAI.

        This private method is called if the `model_name` is "yolo" and a
        `.blob` model path is provided. It sets various parameters specific
        to YOLO models, such as confidence threshold, IOU threshold, anchors,
        and anchor masks, using values from the `self.config` dictionary
        or defaults.

        Requires `self.pipeline` and `self.model_path` to be set.
        """
        if not self.pipeline:
            print("Error: DepthAI pipeline not provided for YOLO model setup.")
            return
        if not self.model_path: # Should be caught by constructor, but defensive check
            print("Error: Model path not set for YOLO model setup.")
            return

        self.nn_node = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        self.nn_node.setBlobPath(str(self.model_path))

        # Set Yolo-specific parameters from config or use defaults
        self.nn_node.setConfidenceThreshold(self.config.get("confidence_threshold", 0.5))
        self.nn_node.setNumClasses(self.config.get("num_classes", 80)) # Common default (COCO)
        self.nn_node.setCoordinateSize(self.config.get("coordinate_size", 4))
        # Default anchors/masks are placeholders; should be specific to the trained model.
        self.nn_node.setAnchors(self.config.get("anchors", [])) # Must be provided for good results
        self.nn_node.setAnchorMasks(self.config.get("anchor_masks", {})) # Must be provided
        self.nn_node.setIouThreshold(self.config.get("iou_threshold", 0.5))

        self.nn_node.input.setBlocking(False) # Common setting for performance
        self.nn_node.setBoundingBoxScaleFactor(self.config.get("bbox_scale_factor", 0.5))
        self.nn_node.setDepthLowerThreshold(self.config.get("depth_lower_thresh", 100)) # In mm
        self.nn_node.setDepthUpperThreshold(self.config.get("depth_upper_thresh", 10000)) # In mm

        self.xout_nn = self.pipeline.create(dai.node.XLinkOut)
        self.xout_nn.setStreamName("detections") # Standard stream name for detections
        self.nn_node.out.link(self.xout_nn.input)

    def _create_generic_nn_network(self) -> None:
        """
        Creates a generic NeuralNetwork node for DepthAI for non-YOLO `.blob` files.

        This is used for models that are not specifically YOLO spatial detection
        networks but are provided as DepthAI `.blob` files.

        Requires `self.pipeline` and `self.model_path` to be set.
        """
        if not self.pipeline:
            print("Error: DepthAI pipeline not provided for generic NN model setup.")
            return
        if not self.model_path:
            print("Error: Model path not set for generic NN model setup.")
            return

        self.nn_node = self.pipeline.create(dai.node.NeuralNetwork)
        self.nn_node.setBlobPath(str(self.model_path))
        self.nn_node.input.setBlocking(False)

        self.xout_nn = self.pipeline.create(dai.node.XLinkOut)
        self.xout_nn.setStreamName("detections") # Or a more generic "nn_out"
        self.nn_node.out.link(self.xout_nn.input)

    def get_model_node(self) -> Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]]:
        """
        Returns the created DepthAI model node.

        This node (either `YoloSpatialDetectionNetwork` or `NeuralNetwork`) should be
        linked to camera inputs in the main pipeline setup.

        Returns:
            Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]]:
                The DepthAI neural network node instance if a DepthAI model was loaded,
                otherwise None.
        """
        return self.nn_node

    def get_output_queues(self, device: dai.Device) -> Dict[str, dai.DataOutputQueue]:
        """
        Gets the output queue for DepthAI neural network results.

        Args:
            device (dai.Device): The DepthAI device instance from which to get the queue.

        Returns:
            Dict[str, dai.DataOutputQueue]: A dictionary containing the output queue for
            detections, keyed by "detections". Empty if no DepthAI model/output is configured.
        """
        queues: Dict[str, dai.DataOutputQueue] = {}
        if self.xout_nn:
            queues["detections"] = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        return queues

    # --- Methods for host-side TFLite model ---
    def _initialize_tflite_detector(self) -> None:
        """
        Initializes a TFLite object detector for host-side processing.

        This method uses the `tflite_support` library to create an `ObjectDetector`
        instance. Configuration parameters like EdgeTPU usage, number of threads,
        score threshold, and max results are taken from `self.config`.

        Requires `self.model_path` to be set to a `.tflite` file.
        Sets `self.tflite_detector` upon success.
        """
        if not self.model_path or not self.model_path.suffix == '.tflite':
            print(f"Warning: Model path '{self.model_path}' is not a .tflite file. "
                  "Cannot initialize TFLite detector.")
            self.tflite_detector = None
            return

        try:
            base_options = core.BaseOptions(
                file_name=str(self.model_path),
                use_coral=self.config.get("enable_edgetpu", False),
                num_threads=self.config.get("num_threads", 4)
            )
            detection_options = processor.DetectionOptions(
                max_results=self.config.get("max_results", 5), # Default from example MLTest
                score_threshold=self.config.get("score_threshold", 0.3) # Default from example MLTest
            )
            options = vision.ObjectDetectorOptions(
                base_options=base_options, detection_options=detection_options)
            self.tflite_detector = vision.ObjectDetector.create_from_options(options)
            print(f"TFLite detector initialized successfully for {self.model_path}")
        except Exception as e: # Broad exception to catch TFLite support errors
            print(f"Error initializing TFLite detector for {self.model_path}: {e}")
            print("Ensure 'tflite_support' library is installed correctly (e.g., 'pip install tflite-support').")
            if self.config.get("enable_edgetpu", False):
                print("For EdgeTPU, also ensure the EdgeTPU runtime library is installed.")
            self.tflite_detector = None

    def detect_with_tflite(self, frame_rgb: np.ndarray) -> Optional[processor.DetectionResult]:
        """
        Performs object detection on an image using the host-side TFLite model.

        Args:
            frame_rgb (np.ndarray): The input image in RGB format (NumPy array).

        Returns:
            Optional[tflite_support.task.processor.DetectionResult]:
                The detection results object from `tflite_support`. Contains a list of
                `Detection` objects. Returns `None` if the `tflite_detector` was not
                initialized or if an error occurs during inference.
        """
        if not self.tflite_detector:
            # print("TFLite detector not initialized. Cannot perform detection.")
            return None

        try:
            # The TFLite Task Library expects an RGB image.
            input_tensor = vision.TensorImage.create_from_array(frame_rgb)
            detection_result = self.tflite_detector.detect(input_tensor)
            return detection_result
        except Exception as e: # Broad exception for runtime errors during detection
            print(f"Error during TFLite detection with model {self.model_path}: {e}")
            return None


if __name__ == '__main__':
    # This main section is for conceptual demonstration and testing of ModelLoader.
    # Full functionality for DepthAI models requires a connected device.
    # Full functionality for TFLite models requires the tflite_support library and a valid model file.

    print("--- ModelLoader Conceptual Examples ---")

    # --- Example 1: YOLO on DepthAI (Conceptual) ---
    print("\n--- Example 1: YOLO on DepthAI (Conceptual) ---")
    mock_pipeline_yolo = dai.Pipeline()
    dummy_yolo_blob_path = Path("dummy_yolo_model.blob")
    dummy_yolo_blob_path.touch() # Create a dummy file for the example

    yolo_model_config = {
        "confidence_threshold": 0.55, "num_classes": 10, "input_size": (416, 416),
        "anchors": [10,13, 16,30, 33,23, 30,61, 62,45, 59,119], # Example anchors
        "anchor_masks": {"side26": [0,1,2], "side13": [3,4,5]} # Example masks
    }
    try:
        yolo_loader_instance = ModelLoader(pipeline=mock_pipeline_yolo, model_name="yolo",
                                           model_path=str(dummy_yolo_blob_path), config=yolo_model_config)
        yolo_model_node = yolo_loader_instance.get_model_node()
        if yolo_model_node:
            print(f"  Successfully created conceptual YOLO NN Node: {type(yolo_model_node)}")
            # print(f"  Confidence Threshold: {yolo_model_node.getConfidenceThreshold()}") # Verify config
        else:
            print("  Failed to create conceptual YOLO NN Node.")
    except FileNotFoundError as e_fnf:
        print(f"  Error: {e_fnf}")
    finally:
        dummy_yolo_blob_path.unlink(missing_ok=True)


    # --- Example 2: Generic Blob Model on DepthAI (Conceptual) ---
    print("\n--- Example 2: Generic Blob Model on DepthAI (Conceptual) ---")
    mock_pipeline_generic = dai.Pipeline()
    dummy_generic_blob_path = Path("dummy_generic_model.blob")
    dummy_generic_blob_path.touch()
    try:
        generic_loader_instance = ModelLoader(pipeline=mock_pipeline_generic, model_name="custom_cnn",
                                              model_path=str(dummy_generic_blob_path))
        generic_model_node = generic_loader_instance.get_model_node()
        if generic_model_node:
            print(f"  Successfully created conceptual Generic NN Node: {type(generic_model_node)}")
        else:
            print("  Failed to create conceptual Generic NN Node.")
    except FileNotFoundError as e_fnf:
        print(f"  Error: {e_fnf}")
    finally:
        dummy_generic_blob_path.unlink(missing_ok=True)


    # --- Example 3: TFLite Host Model ---
    print("\n--- Example 3: TFLite Host Model (Conceptual) ---")
    dummy_tflite_model_path = Path("dummy_efficientdet_lite0.tflite")
    # A real .tflite model file is needed for tflite_support to load it without error.
    # Touching an empty file will likely cause an error during _initialize_tflite_detector.
    # For a true test, replace with a path to a valid .tflite model.
    dummy_tflite_model_path.touch() # Creates an empty file

    tflite_model_config = {
        "enable_edgetpu": False, "num_threads": 2,
        "score_threshold": 0.6, "max_results": 3
    }
    tflite_loader_instance = None
    try:
        tflite_loader_instance = ModelLoader(model_name="efficientdet_lite0_tflite",
                                             model_path=str(dummy_tflite_model_path),
                                             config=tflite_model_config)
        if tflite_loader_instance.tflite_detector:
            print(f"  Successfully initialized TFLite detector for: {tflite_loader_instance.model_path}")
            # Create a dummy RGB frame for conceptual detection
            dummy_frame_h, dummy_frame_w = 300, 300 # Example dimensions
            if "input_size" in tflite_model_config and tflite_model_config["input_size"]:
                 dummy_frame_h, dummy_frame_w = tflite_model_config["input_size"]
            dummy_rgb_image = np.zeros((dummy_frame_h, dummy_frame_w, 3), dtype=np.uint8)

            print(f"  Attempting conceptual detection with dummy RGB frame (shape: {dummy_rgb_image.shape})...")
            detection_output = tflite_loader_instance.detect_with_tflite(dummy_rgb_image)

            if detection_output is not None and hasattr(detection_output, 'detections'):
                if detection_output.detections: # Check if list is not empty
                    print(f"  Conceptual detection result (example): {detection_output.detections[0]}")
                else:
                    print("  Conceptual TFLite detection ran, but no objects found (as expected with a dummy model).")
            else: # detection_output is None or does not have 'detections'
                print("  Conceptual TFLite detection failed or returned None (expected with a non-functional dummy .tflite file).")
        else:
            model_path_str = str(tflite_loader_instance.model_path) if tflite_loader_instance and tflite_loader_instance.model_path else str(dummy_tflite_model_path)
            print(f"  Failed to initialize TFLite detector for {model_path_str} (this is expected if the .tflite file is a dummy/empty file).")

    except FileNotFoundError as e_fnf:
        print(f"  Error: {e_fnf}")
    except Exception as e_main: # Catch any other unexpected errors
        print(f"  An unexpected error occurred in TFLite example: {e_main}")
    finally:
        if dummy_tflite_model_path.exists():
            dummy_tflite_model_path.unlink()
            print(f"  Dummy TFLite file {dummy_tflite_model_path} cleaned up.")


    print("\n--- ModelLoader conceptual examples complete. ---")
    print("Note: For actual operation, valid model files and, for DepthAI, a connected device are necessary.")

```
