"""
DepthAI Camera Management Module.

This module provides the `CameraManager` class, which is responsible for
initializing and configuring the camera system on a DepthAI device. It handles
the setup of RGB cameras, mono cameras for stereo depth, and the stereo depth
node itself. It also manages the creation of output streams for camera data
and provides linking capabilities to neural network model inputs.
"""
import depthai as dai
from typing import Tuple, Dict, Optional, Any # For type hinting

class CameraManager:
    """
    Manages DepthAI camera initialization, configuration, and frame retrieval.

    This class abstracts the complexities of setting up various camera nodes
    (RGB, mono) and the stereo depth perception node in a DepthAI pipeline.
    It configures camera properties like preview size, resolution, and color order,
    and sets up XLinkOut streams for RGB and depth data.

    Attributes:
        pipeline (dai.Pipeline): The DepthAI pipeline object. All camera nodes
            are added to this pipeline.
        preview_size (Tuple[int, int]): The (width, height) of the RGB camera preview.
        nn_input_size (Tuple[int, int]): The (width, height) expected by the neural
            network. This might differ from `preview_size`.
        model_name (str): Name of the model being used (e.g., "yolo", "mobilenet").
            This influences how camera outputs are linked to the model.
        cam_rgb (Optional[dai.node.ColorCamera]): The RGB camera node.
        mono_left (Optional[dai.node.MonoCamera]): The left mono camera node.
        mono_right (Optional[dai.node.MonoCamera]): The right mono camera node.
        stereo (Optional[dai.node.StereoDepth]): The stereo depth perception node.
        xout_rgb (Optional[dai.node.XLinkOut]): Output stream for RGB data.
        xout_depth (Optional[dai.node.XLinkOut]): Output stream for depth data.
        xout_nn_passthrough (Optional[dai.node.XLinkOut]): Output stream for RGB data
            that has passed through a neural network node (e.g., YOLO passthrough).
    """
    def __init__(self,
                 preview_size: Tuple[int, int] = (640, 640),
                 nn_input_size: Optional[Tuple[int, int]] = None,
                 model_name: str = "yolo"):
        """
        Initializes the CameraManager.

        Sets up the DepthAI pipeline and calls internal methods to create
        RGB and stereo camera nodes.

        Args:
            preview_size (Tuple[int, int]): Size (width, height) of the RGB camera
                preview. Defaults to (640, 640).
            nn_input_size (Optional[Tuple[int, int]]): Size (width, height) of the input
                expected by the neural network. If None, it defaults to `preview_size`.
                Defaults to None.
            model_name (str): Name of the model type (e.g., "yolo", "mobilenet").
                This helps determine how camera outputs are linked to the model,
                especially for models like YOLO that have passthrough streams.
                Defaults to "yolo".
        """
        self.pipeline: dai.Pipeline = dai.Pipeline()
        self.preview_size: Tuple[int, int] = preview_size
        self.nn_input_size: Tuple[int, int] = nn_input_size if nn_input_size else preview_size
        self.model_name: str = model_name.lower()

        self.cam_rgb: Optional[dai.node.ColorCamera] = None
        self.mono_left: Optional[dai.node.MonoCamera] = None
        self.mono_right: Optional[dai.node.MonoCamera] = None
        self.stereo: Optional[dai.node.StereoDepth] = None

        self.xout_rgb: Optional[dai.node.XLinkOut] = None
        self.xout_depth: Optional[dai.node.XLinkOut] = None
        # For YOLO, RGB can come from NN passthrough; for others, directly from cam_rgb
        self.xout_nn_passthrough: Optional[dai.node.XLinkOut] = None

        self._create_rgb_camera()
        self._create_stereo_cameras_and_depth()

    def _create_rgb_camera(self) -> None:
        """
        Creates and configures the RGB color camera node.

        This private method initializes the `ColorCamera` node, sets its preview size,
        resolution, color order (BGR for OpenCV compatibility), and configures an
        XLinkOut stream for the RGB data. If the model is not "yolo", it links
        the camera's preview output directly to this RGB XLinkOut stream.
        """
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        if self.cam_rgb is None: # Should not happen with create
            raise RuntimeError("Failed to create ColorCamera node.")
            
        self.cam_rgb.setPreviewSize(self.preview_size[0], self.preview_size[1])
        # Higher resolution for still images or if ISP scaling is desired for preview
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setInterleaved(False) # Output planar data
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        if self.xout_rgb is None:
            raise RuntimeError("Failed to create XLinkOut for RGB.")
        self.xout_rgb.setStreamName("rgb")

        # For non-YOLO models, link cam_rgb.preview directly to xout_rgb.
        # For YOLO, RGB data for display often comes from the YoloSpatialDetectionNetwork's
        # passthrough output, which is handled in `link_model_inputs`.
        if self.model_name != "yolo":
             self.cam_rgb.preview.link(self.xout_rgb.input)


    def _create_stereo_cameras_and_depth(self) -> None:
        """
        Creates and configures mono cameras and the stereo depth node.

        This method initializes the left and right `MonoCamera` nodes and the
        `StereoDepth` node. It configures mono camera resolution, sets stereo depth
        parameters like preset mode, alignment, output size, and subpixel disparity.
        It also links the mono camera outputs to the stereo depth node inputs and
        sets up an XLinkOut stream for depth data. If the model is not "yolo",
        the stereo depth output is linked directly to this depth XLinkOut stream.
        """
        self.mono_left = self.pipeline.create(dai.node.MonoCamera)
        if self.mono_left is None: raise RuntimeError("Failed to create MonoCamera (left).")
        self.mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_left.setCamera("left")

        self.mono_right = self.pipeline.create(dai.node.MonoCamera)
        if self.mono_right is None: raise RuntimeError("Failed to create MonoCamera (right).")
        self.mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_right.setCamera("right")

        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        if self.stereo is None: raise RuntimeError("Failed to create StereoDepth node.")
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of the RGB camera for easier fusion
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # CAM_A is typically RGB camera
        # Output size of depth map can be configured; here matches mono camera resolution
        self.stereo.setOutputSize(self.mono_left.getResolutionWidth(), self.mono_left.getResolutionHeight())
        self.stereo.setSubpixel(True) # Enable subpixel disparity for more precise depth

        # Link mono cameras to stereo depth node
        self.mono_left.out.link(self.stereo.left)
        self.mono_right.out.link(self.stereo.right)

        self.xout_depth = self.pipeline.create(dai.node.XLinkOut)
        if self.xout_depth is None: raise RuntimeError("Failed to create XLinkOut for depth.")
        self.xout_depth.setStreamName("depth")

        # For non-YOLO models, link stereo.depth directly to xout_depth.
        # For YOLO, depth data is typically an input to the YoloSpatialDetectionNetwork,
        # and the processed depth (or passthrough depth) is output from there.
        if self.model_name != "yolo":
            self.stereo.depth.link(self.xout_depth.input)


    def link_model_inputs(self, model_node: Any) -> None:
        """
        Links camera outputs to the provided model node's inputs.

        This method should be called after the model node (e.g., from `ModelLoader`)
        has been created and before the pipeline is started. The linking strategy
        depends on the `model_name` and the type of `model_node`.

        Args:
            model_node (Any): The neural network node instance (e.g.,
                `dai.node.YoloSpatialDetectionNetwork`, `dai.node.MobileNetDetectionNetwork`,
                `dai.node.NeuralNetwork`). The type hint `Any` is used for flexibility
                with different DepthAI model node types.

        Raises:
            TypeError: If `model_node` is None.
            RuntimeError: If critical components like `cam_rgb` or `stereo` are not initialized.
        """
        if model_node is None:
            raise TypeError("model_node cannot be None for linking.")
        if self.cam_rgb is None or self.stereo is None:
             raise RuntimeError("RGB Camera or StereoDepth node not initialized before linking.")

        if self.model_name == "yolo" and isinstance(model_node, dai.node.YoloSpatialDetectionNetwork):
            self.cam_rgb.preview.link(model_node.input)
            self.stereo.depth.link(model_node.inputDepth)

            # Create XLinkOut for RGB passthrough from YOLO node (for synchronized display)
            self.xout_nn_passthrough = self.pipeline.create(dai.node.XLinkOut)
            if self.xout_nn_passthrough is None:
                raise RuntimeError("Failed to create XLinkOut for NN passthrough.")
            self.xout_nn_passthrough.setStreamName("rgb_passthrough_nn")
            model_node.passthrough.link(self.xout_nn_passthrough.input)

            # For YOLO, depth output also comes from its passthroughDepth stream
            if self.xout_depth: # Ensure xout_depth was created
                model_node.passthroughDepth.link(self.xout_depth.input)
            else:
                raise RuntimeError("xout_depth is None, cannot link YOLO passthroughDepth.")


        elif (self.model_name in ["mobilenet", "efficientdet"] or isinstance(model_node, dai.node.NeuralNetwork)) \
             and not isinstance(model_node, dai.node.YoloSpatialDetectionNetwork):
            # For generic NeuralNetwork or specific non-YOLO detection networks
            # Link RGB camera preview to model input.
            # An ImageManip node might be needed if nn_input_size differs significantly from
            # preview_size and the model doesn't handle resizing/cropping well.
            # For simplicity, assuming preview is suitable or an ImageManip is added elsewhere.
            self.cam_rgb.preview.link(model_node.input)

            # If the model requires depth as an input (e.g., some custom NNs)
            if hasattr(model_node, 'inputDepth') and model_node.inputDepth is not None:
                 self.stereo.depth.link(model_node.inputDepth)
            # If this generic model doesn't use depth as input, then stereo.depth should be
            # linked to self.xout_depth if it wasn't already (e.g. if model_name was not "yolo")
            # This logic is handled in _create_stereo_cameras_and_depth based on model_name.
        else:
            print(f"Warning: Model type '{self.model_name}' or node type '{type(model_node)}' "
                  "has unhandled input linking logic. Ensure manual linking if necessary.")


    def get_output_queues(self, device: dai.Device) -> Dict[str, dai.DataOutputQueue]:
        """
        Gets the output queues for RGB and depth data streams from the device.

        The specific RGB queue returned depends on the `model_name`. For "yolo" models
        with a passthrough stream, the RGB queue is taken from `xout_nn_passthrough`.
        Otherwise, it's from the direct `xout_rgb`.

        Args:
            device (dai.Device): The DepthAI device instance.

        Returns:
            Dict[str, dai.DataOutputQueue]: A dictionary containing the output queues.
            Keys typically include "rgb" and "depth".
        """
        queues: Dict[str, dai.DataOutputQueue] = {}

        # Determine the source of the RGB queue
        if self.model_name == "yolo" and self.xout_nn_passthrough:
            # For YOLO, the RGB feed for display often comes from the NN passthrough
            # to ensure synchronization with detections.
            queues["rgb"] = device.getOutputQueue(name="rgb_passthrough_nn", maxSize=4, blocking=False)
        elif self.xout_rgb: # For other models or if YOLO passthrough isn't used for main RGB
            queues["rgb"] = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        if self.xout_depth:
            queues["depth"] = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        return queues

if __name__ == '__main__':
    # This section is for conceptual demonstration.
    # Running it requires a connected DepthAI device.
    print("--- CameraManager Conceptual Example ---")
    try:
        # Example 1: CameraManager for a YOLO model
        print("\nInitializing CameraManager for YOLO model...")
        yolo_camera_manager = CameraManager(preview_size=(416, 416), model_name="yolo")
        print("  YOLO CameraManager pipeline nodes (before linking model):")
        for node in yolo_camera_manager.pipeline.getAllNodes(): # type: ignore
            print(f"    Node: {node.getName()}, Type: {type(node)}")
        # In a real app, you'd get a model_node from ModelLoader and call:
        # mock_yolo_node = yolo_camera_manager.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        # yolo_camera_manager.link_model_inputs(mock_yolo_node)


        # Example 2: CameraManager for a generic model (e.g., MobileNet)
        print("\nInitializing CameraManager for a generic MobileNet-like model...")
        generic_camera_manager = CameraManager(preview_size=(300, 300),
                                               nn_input_size=(300,300),
                                               model_name="mobilenet")
        print("  Generic Model CameraManager pipeline nodes (before linking model):")
        for node in generic_camera_manager.pipeline.getAllNodes(): # type: ignore
            print(f"    Node: {node.getName()}, Type: {type(node)}")
        # mock_generic_node = generic_camera_manager.pipeline.create(dai.node.NeuralNetwork)
        # generic_camera_manager.link_model_inputs(mock_generic_node)


        print("\nCameraManager basic setup examples complete.")
        print("Full test requires a DepthAI device and integration with ModelLoader.")

    except Exception as e_main_example:
        print(f"Error during CameraManager example: {e_main_example}")

    print("\nNote on typical full pipeline integration:")
    print("1. `cam_manager = CameraManager(model_name='your_model_type', ...)`")
    print("2. `model_loader = ModelLoader(pipeline=cam_manager.pipeline, ...)`")
    print("3. `model_node = model_loader.get_model_node()`")
    print("4. `cam_manager.link_model_inputs(model_node)`")
    print("5. `with dai.Device(cam_manager.pipeline) as device:`")
    print("   `  rgb_q = cam_manager.get_output_queues(device)['rgb']`")
    print("   `  det_q = model_loader.get_output_queues(device)['detections']`")
    print("   `  # ... main loop processing queues ...`")

```
