# Module src.camera_manager

> DepthAI Camera Management Module.
>
> This module provides the `CameraManager` class, which is responsible for
> initializing and configuring the camera system on a DepthAI device. It handles
> the setup of RGB cameras, mono cameras for stereo depth, and the stereo depth
> node itself. It also manages the creation of output streams for camera data
> and provides linking capabilities to neural network model inputs.

---
## Class CameraManager

Manages DepthAI camera initialization, configuration, and frame retrieval.

This class abstracts the complexities of setting up various camera nodes
(RGB, mono) and the stereo depth perception node in a DepthAI pipeline.
It configures camera properties like preview size, resolution, and color order,
and sets up XLinkOut streams for RGB and depth data.

#### Attributes
*   **pipeline** (*dai.Pipeline*): The DepthAI pipeline object. All camera nodes are added to this pipeline.
*   **preview_size** (*Tuple[int, int]*): The (width, height) of the RGB camera preview.
*   **nn_input_size** (*Tuple[int, int]*): The (width, height) expected by the neural network. This might differ from `preview_size`.
*   **model_name** (*str*): Name of the model being used (e.g., "yolo", "mobilenet"). This influences how camera outputs are linked to the model.
*   **cam_rgb** (*Optional[dai.node.ColorCamera]*): The RGB camera node.
*   **mono_left** (*Optional[dai.node.MonoCamera]*): The left mono camera node.
*   **mono_right** (*Optional[dai.node.MonoCamera]*): The right mono camera node.
*   **stereo** (*Optional[dai.node.StereoDepth]*): The stereo depth perception node.
*   **xout_rgb** (*Optional[dai.node.XLinkOut]*): Output stream for RGB data.
*   **xout_depth** (*Optional[dai.node.XLinkOut]*): Output stream for depth data.
*   **xout_nn_passthrough** (*Optional[dai.node.XLinkOut]*): Output stream for RGB data that has passed through a neural network node (e.g., YOLO passthrough).

### Methods
*   `__init__`
*   `_create_rgb_camera`
*   `_create_stereo_cameras_and_depth`
*   `link_model_inputs`
*   `get_output_queues`

---
### `__init__`
```python
def __init__(self, preview_size: Tuple[int, int] = (640, 640), nn_input_size: Optional[Tuple[int, int]] = None, model_name: str = "yolo"):
```
Initializes the CameraManager.

Sets up the DepthAI pipeline and calls internal methods to create
RGB and stereo camera nodes.

#### Arguments
*   **preview_size** (*Tuple[int, int], optional*): Size (width, height) of the RGB camera preview. Defaults to (640, 640).
*   **nn_input_size** (*Optional[Tuple[int, int]], optional*): Size (width, height) of the input expected by the neural network. If None, it defaults to `preview_size`. Defaults to None.
*   **model_name** (*str, optional*): Name of the model type (e.g., "yolo", "mobilenet"). This helps determine how camera outputs are linked to the model, especially for models like YOLO that have passthrough streams. Defaults to "yolo".

---
### `_create_rgb_camera`
```python
def _create_rgb_camera(self) -> None:
```
Creates and configures the RGB color camera node.

This private method initializes the `ColorCamera` node, sets its preview size,
resolution, color order (BGR for OpenCV compatibility), and configures an
XLinkOut stream for the RGB data. If the model is not "yolo", it links
the camera's preview output directly to this RGB XLinkOut stream.

---
### `_create_stereo_cameras_and_depth`
```python
def _create_stereo_cameras_and_depth(self) -> None:
```
Creates and configures mono cameras and the stereo depth node.

This method initializes the left and right `MonoCamera` nodes and the
`StereoDepth` node. It configures mono camera resolution, sets stereo depth
parameters like preset mode, alignment, output size, and subpixel disparity.
It also links the mono camera outputs to the stereo depth node inputs and
sets up an XLinkOut stream for depth data. If the model is not "yolo",
the stereo depth output is linked directly to this depth XLinkOut stream.

---
### `link_model_inputs`
```python
def link_model_inputs(self, model_node: Any) -> None:
```
Links camera outputs to the provided model node's inputs.

This method should be called after the model node (e.g., from `ModelLoader`)
has been created and before the pipeline is started. The linking strategy
depends on the `model_name` and the type of `model_node`.

#### Arguments
*   **model_node** (*Any*): The neural network node instance (e.g., `dai.node.YoloSpatialDetectionNetwork`, `dai.node.MobileNetDetectionNetwork`, `dai.node.NeuralNetwork`). The type hint `Any` is used for flexibility with different DepthAI model node types.

#### Raises
*   **TypeError**: If `model_node` is None.
*   **RuntimeError**: If critical components like `cam_rgb` or `stereo` are not initialized.

---
### `get_output_queues`
```python
def get_output_queues(self, device: dai.Device) -> Dict[str, dai.DataOutputQueue]:
```
Gets the output queues for RGB and depth data streams from the device.

The specific RGB queue returned depends on the `model_name`. For "yolo" models
with a passthrough stream, the RGB queue is taken from `xout_nn_passthrough`.
Otherwise, it's from the direct `xout_rgb`.

#### Arguments
*   **device** (*dai.Device*): The DepthAI device instance.

#### Returns
*   (*Dict[str, dai.DataOutputQueue]*): A dictionary containing the output queues. Keys typically include "rgb" and "depth".

---
```
