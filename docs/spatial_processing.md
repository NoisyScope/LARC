# Module src.spatial_processing

> Spatial Processing Module for DepthAI.
>
> This module provides the `SpatialProcessor` class, which is responsible for
> handling depth frames from DepthAI cameras and extracting spatial information
> from object detections. This includes colorizing depth maps for visualization
> and calculating real-world X, Y, Z coordinates of detected objects.

---
## Class SpatialProcessor

Handles processing of depth frames and extraction of spatial coordinates from detections.

This class provides methods to convert raw depth data into a more human-readable
format (e.g., a colorized depth map) and to extract 3D spatial coordinates
(X, Y, Z) and 2D bounding box information from DepthAI detection objects.

#### Attributes
*   **default_color_map** (*int*): The OpenCV colormap (e.g., `cv2.COLORMAP_HOT`) used for visualizing depth frames.

### Methods
*   `__init__`
*   `process_depth_frame`
*   `get_spatial_coordinates`
*   `get_roi_from_detection`

---
### `__init__`
```python
def __init__(self, default_color_map: int = cv2.COLORMAP_HOT):
```
Initializes the SpatialProcessor.

#### Arguments
*   **default_color_map** (*int, optional*): The default OpenCV colormap to apply to depth frames for visualization. Defaults to `cv2.COLORMAP_HOT`.

---
### `process_depth_frame`
```python
def process_depth_frame(self, depth_frame: np.ndarray, downscale_factor: int = 4) -> tuple[np.ndarray | None, np.ndarray | None]:
```
Processes a raw depth frame for visualization.

This method takes a raw depth frame (typically with values in millimeters),
optionally downscales it for percentile calculation to determine min/max depth,
normalizes the depth values, and applies a colormap for visualization.

#### Arguments
*   **depth_frame** (*np.ndarray*): The raw depth frame, usually a 2D NumPy array where values represent depth in millimeters.
*   **downscale_factor** (*int, optional*): The factor by which to downscale the depth frame for percentile calculation. This helps in getting a more robust min/max range by ignoring extreme outliers. If 1 or less, no downscaling is performed for percentile calculation. Defaults to 4.

#### Returns
*   *tuple[np.ndarray | None, np.ndarray | None]*: 
    *   The colorized depth frame (np.ndarray) ready for display. Returns None if the input `depth_frame` is None.
    *   The original depth frame (np.ndarray, or a copy). Returns None if the input `depth_frame` is None.

---
### `get_spatial_coordinates`
```python
def get_spatial_coordinates(self, detection: 'any', frame_width: int = None, frame_height: int = None) -> tuple[float, float, float, float, float, float, float]:
```
Extracts spatial coordinates (X, Y, Z) from a DepthAI detection object.

Also returns bounding box coordinates. If `frame_width` and `frame_height`
are provided, the bounding box coordinates are denormalized to pixel values.
Otherwise, normalized coordinates (0-1 range) are returned.

#### Arguments
*   **detection** (*any*): A DepthAI detection object. For spatial detections (e.g., from `YoloSpatialDetectionNetwork`), this object is expected to have a `spatialCoordinates` attribute with `x`, `y`, `z` fields (typically in millimeters), and `xmin`, `xmax`, `ymin`, `ymax` attributes for the normalized bounding box.
*   **frame_width** (*int, optional*): Width of the video frame in pixels. Used for denormalizing bounding box coordinates.
*   **frame_height** (*int, optional*): Height of the video frame in pixels. Used for denormalizing bounding box coordinates.

#### Returns
*   *tuple[float, float, float, float, float, float, float]*: A tuple containing:
    *   x_coord (float): Spatial X coordinate (e.g., in mm).
    *   y_coord (float): Spatial Y coordinate (e.g., in mm).
    *   z_coord (float): Spatial Z coordinate (depth, e.g., in mm).
    *   xmin (float): Minimum X coordinate of the bounding box. Pixel value if `frame_width` provided, else normalized.
    *   ymin (float): Minimum Y coordinate of the bounding box. Pixel value if `frame_height` provided, else normalized.
    *   xmax (float): Maximum X coordinate of the bounding box. Pixel value if `frame_width` provided, else normalized.
    *   ymax (float): Maximum Y coordinate of the bounding box. Pixel value if `frame_height` provided, else normalized.

#### Raises
*   **AttributeError**: If the `detection` object does not have the expected `spatialCoordinates` or bounding box attributes.

---
### `get_roi_from_detection`
```python
def get_roi_from_detection(self, detection: 'any', depth_frame_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
```
Extracts the Region of Interest (ROI) pixel coordinates from a detection's `boundingBoxMapping`.

This method is typically used when detections provide ROI information relative
to a specific frame (often the depth frame), and that ROI needs to be
denormalized to pixel coordinates.

#### Arguments
*   **detection** (*any*): A DepthAI detection object. Expected to have a `boundingBoxMapping` attribute, which in turn has an `roi` attribute. The `roi` attribute should be an object with a `denormalize` method (like `depthai.Rect`) that accepts frame width and height.
*   **depth_frame_shape** (*tuple[int, int]*): Shape of the depth frame (height, width) used for denormalizing the ROI coordinates.

#### Returns
*   *tuple[int, int, int, int] | None*: A tuple `(xmin, ymin, xmax, ymax)` representing the pixel coordinates of the ROI in the depth frame. Returns `None` if `boundingBoxMapping` or `roi` is not available in the detection object, or if denormalization fails.

#### Raises
*   **AttributeError**: If `detection.boundingBoxMapping.roi` does not have a `denormalize` method or `topLeft`/`bottomRight` methods after denormalization.

---
```
