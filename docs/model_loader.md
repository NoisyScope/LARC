# Module src.model_loader

> Machine Learning Model Loader for DepthAI and Host Processing.
>
> This module provides the `ModelLoader` class, designed to handle the
> loading and configuration of machine learning models for both DepthAI
> on-device inference (using `.blob` files) and host-side inference
> (using `.tflite` files with TensorFlow Lite Runtime/Support).
>
> It supports different model architectures like YOLO for spatial detection
> on DepthAI, generic neural network blobs for DepthAI, and TFLite models
> for execution on the host CPU or EdgeTPU.

---
## Class ModelLoader

Manages loading and running ML models on DepthAI or the host.

This class encapsulates the logic for initializing different types of
neural network models, whether they are DepthAI-specific blob files
or TensorFlow Lite models for host-side execution. It configures
the model based on provided parameters and prepares it for inference.

#### Attributes
*   **pipeline** (*Optional[dai.Pipeline]*): The DepthAI pipeline object. Required for DepthAI blob models to add the NN node.
*   **model_name** (*str*): The name or type of the model (e.g., "yolo", "efficientdet_tflite").
*   **model_path** (*Optional[Path]*): The file path to the model (.blob or .tflite).
*   **config** (*Dict[str, Any]*): Configuration dictionary for model-specific parameters.
*   **nn_node** (*Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]]*): The DepthAI neural network node instance, if a DepthAI model is loaded.
*   **xout_nn** (*Optional[dai.node.XLinkOut]*): The XLinkOut node for DepthAI model's detection results.
*   **tflite_detector** (*Optional[vision.ObjectDetector]*): The TFLite ObjectDetector instance from `tflite_support`, if a TFLite model is loaded for host processing.

### Methods
*   `__init__`
*   `_create_yolo_spatial_network`
*   `_create_generic_nn_network`
*   `get_model_node`
*   `get_output_queues`
*   `_initialize_tflite_detector`
*   `detect_with_tflite`

---
### `__init__`
```python
def __init__(self, pipeline: Optional[dai.Pipeline] = None, model_name: str = "unknown", model_path: Optional[Union[str, Path]] = None, config: Optional[Dict[str, Any]] = None):
```
Initializes the ModelLoader.

Based on the `model_name`, `model_path` suffix, and whether a `pipeline`
is provided, this constructor will attempt to initialize either a DepthAI
neural network node or a host-side TFLite detector.

#### Arguments
*   **pipeline** (*Optional[dai.Pipeline], optional*): The DepthAI pipeline object to which a DepthAI NN node should be added. Required for `.blob` models. Defaults to None.
*   **model_name** (*str, optional*): Name/type of the model. Used to determine setup logic. Examples: "yolo", "mobilenet_ssd", "efficientdet_tflite". Defaults to "unknown".
*   **model_path** (*Optional[Union[str, Path]], optional*): Path to the model file. Can be a `.blob` for DepthAI or `.tflite` for host processing. Defaults to None.
*   **config** (*Optional[Dict[str, Any]], optional*): Model-specific configurations. See class documentation for examples for YOLO and TFLite. Defaults to an empty dictionary.

#### Raises
*   **FileNotFoundError**: If `model_path` is provided but the file does not exist.

#### Examples
Example Config for YOLO (DepthAI):
```
{
    "confidence_threshold": 0.5, "iou_threshold": 0.5,
    "num_classes": 80, "coordinate_size": 4,
    "anchors": [...], "anchor_masks": {...},
    "input_size": (416, 416) # (width, height)
}
```

Example Config for TFLite (host):
```
{
    "enable_edgetpu": False, "num_threads": 4,
    "score_threshold": 0.5, "max_results": 5,
    "input_size": (300, 300) # (width, height), optional
}
```

---
### `_create_yolo_spatial_network`
```python
def _create_yolo_spatial_network(self) -> None:
```
Creates and configures a YoloSpatialDetectionNetwork node for DepthAI.

This private method is called if the `model_name` is "yolo" and a
`.blob` model path is provided. It sets various parameters specific
to YOLO models, such as confidence threshold, IOU threshold, anchors,
and anchor masks, using values from the `self.config` dictionary
or defaults.

Requires `self.pipeline` and `self.model_path` to be set.

---
### `_create_generic_nn_network`
```python
def _create_generic_nn_network(self) -> None:
```
Creates a generic NeuralNetwork node for DepthAI for non-YOLO `.blob` files.

This is used for models that are not specifically YOLO spatial detection
networks but are provided as DepthAI `.blob` files.

Requires `self.pipeline` and `self.model_path` to be set.

---
### `get_model_node`
```python
def get_model_node(self) -> Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]]:
```
Returns the created DepthAI model node.

This node (either `YoloSpatialDetectionNetwork` or `NeuralNetwork`) should be
linked to camera inputs in the main pipeline setup.

#### Returns
*   (*Optional[Union[dai.node.YoloSpatialDetectionNetwork, dai.node.NeuralNetwork]]*): The DepthAI neural network node instance if a DepthAI model was loaded, otherwise None.

---
### `get_output_queues`
```python
def get_output_queues(self, device: dai.Device) -> Dict[str, dai.DataOutputQueue]:
```
Gets the output queue for DepthAI neural network results.

#### Arguments
*   **device** (*dai.Device*): The DepthAI device instance from which to get the queue.

#### Returns
*   (*Dict[str, dai.DataOutputQueue]*): A dictionary containing the output queue for detections, keyed by "detections". Empty if no DepthAI model/output is configured.

---
### `_initialize_tflite_detector`
```python
def _initialize_tflite_detector(self) -> None:
```
Initializes a TFLite object detector for host-side processing.

This method uses the `tflite_support` library to create an `ObjectDetector`
instance. Configuration parameters like EdgeTPU usage, number of threads,
score threshold, and max results are taken from `self.config`.

Requires `self.model_path` to be set to a `.tflite` file.
Sets `self.tflite_detector` upon success.

---
### `detect_with_tflite`
```python
def detect_with_tflite(self, frame_rgb: np.ndarray) -> Optional[processor.DetectionResult]:
```
Performs object detection on an image using the host-side TFLite model.

#### Arguments
*   **frame_rgb** (*np.ndarray*): The input image in RGB format (NumPy array).

#### Returns
*   (*Optional[tflite_support.task.processor.DetectionResult]*): The detection results object from `tflite_support`. Contains a list of `Detection` objects. Returns `None` if the `tflite_detector` was not initialized or if an error occurs during inference.

---
```
