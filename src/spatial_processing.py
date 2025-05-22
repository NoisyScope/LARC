"""
Spatial Processing Module for DepthAI.

This module provides the `SpatialProcessor` class, which is responsible for
handling depth frames from DepthAI cameras and extracting spatial information
from object detections. This includes colorizing depth maps for visualization
and calculating real-world X, Y, Z coordinates of detected objects.
"""
import numpy as np
import cv2

class SpatialProcessor:
    """
    Handles processing of depth frames and extraction of spatial coordinates from detections.

    This class provides methods to convert raw depth data into a more human-readable
    format (e.g., a colorized depth map) and to extract 3D spatial coordinates
    (X, Y, Z) and 2D bounding box information from DepthAI detection objects.

    Attributes:
        default_color_map (int): The OpenCV colormap (e.g., `cv2.COLORMAP_HOT`)
            used for visualizing depth frames.
    """
    def __init__(self, default_color_map: int = cv2.COLORMAP_HOT):
        """
        Initializes the SpatialProcessor.

        Args:
            default_color_map (int): The default OpenCV colormap to apply to depth frames
                for visualization. Defaults to `cv2.COLORMAP_HOT`.
        """
        self.default_color_map = default_color_map

    def process_depth_frame(self, depth_frame: np.ndarray, downscale_factor: int = 4) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Processes a raw depth frame for visualization.

        This method takes a raw depth frame (typically with values in millimeters),
        optionally downscales it for percentile calculation to determine min/max depth,
        normalizes the depth values, and applies a colormap for visualization.

        Args:
            depth_frame (np.ndarray): The raw depth frame, usually a 2D NumPy array
                where values represent depth in millimeters.
            downscale_factor (int): The factor by which to downscale the depth frame
                for percentile calculation. This helps in getting a more robust
                min/max range by ignoring extreme outliers. If 1 or less,
                no downscaling is performed for percentile calculation. Defaults to 4.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]:
                - The colorized depth frame (np.ndarray) ready for display.
                  Returns None if the input `depth_frame` is None.
                - The original depth frame (np.ndarray, or a copy).
                  Returns None if the input `depth_frame` is None.
        """
        if depth_frame is None:
            return None, None

        depth_frame_copy = depth_frame.copy()

        if downscale_factor and downscale_factor > 1:
            depth_downscaled = depth_frame_copy[::downscale_factor, ::downscale_factor]
        else:
            depth_downscaled = depth_frame_copy

        # Handle cases where depth_downscaled might be all zeros (e.g., if camera is blocked)
        if np.all(depth_downscaled == 0):
            min_depth = 0.0
            max_depth = 1.0 # Avoid division by zero if min_depth is also 0
        else:
            # Calculate robust min and max depth using percentiles, ignoring zeros
            non_zero_depths = depth_downscaled[depth_downscaled != 0]
            if non_zero_depths.size > 0:
                min_depth = np.percentile(non_zero_depths, 1)
                max_depth = np.percentile(non_zero_depths, 99)
            else: # If all non-downscaled values were zero
                min_depth = 0.0
                max_depth = 1.0

        # Ensure max_depth is greater than min_depth to avoid issues with interp
        if max_depth <= min_depth:
            max_depth = min_depth + 1.0 # Arbitrary small difference

        # Normalize and colorize the original full-resolution depth frame
        # np.interp requires float inputs for x and xp
        depth_frame_normalized = np.interp(depth_frame_copy.astype(float),
                                           (min_depth, max_depth),
                                           (0, 255)).astype(np.uint8)
        depth_frame_colorized = cv2.applyColorMap(depth_frame_normalized, self.default_color_map)

        return depth_frame_colorized, depth_frame_copy


    def get_spatial_coordinates(self, detection: 'any', frame_width: int = None, frame_height: int = None) -> tuple[float, float, float, float, float, float, float]:
        """
        Extracts spatial coordinates (X, Y, Z) from a DepthAI detection object.

        Also returns bounding box coordinates. If `frame_width` and `frame_height`
        are provided, the bounding box coordinates are denormalized to pixel values.
        Otherwise, normalized coordinates (0-1 range) are returned.

        Args:
            detection (any): A DepthAI detection object. For spatial detections
                (e.g., from `YoloSpatialDetectionNetwork`), this object is expected
                to have a `spatialCoordinates` attribute with `x`, `y`, `z` fields
                (typically in millimeters), and `xmin`, `xmax`, `ymin`, `ymax` attributes
                for the normalized bounding box.
            frame_width (int, optional): Width of the video frame in pixels.
                Used for denormalizing bounding box coordinates.
            frame_height (int, optional): Height of the video frame in pixels.
                Used for denormalizing bounding box coordinates.

        Returns:
            tuple[float, float, float, float, float, float, float]: A tuple containing:
                - x_coord (float): Spatial X coordinate (e.g., in mm).
                - y_coord (float): Spatial Y coordinate (e.g., in mm).
                - z_coord (float): Spatial Z coordinate (depth, e.g., in mm).
                - xmin (float): Minimum X coordinate of the bounding box.
                                Pixel value if `frame_width` provided, else normalized.
                - ymin (float): Minimum Y coordinate of the bounding box.
                                Pixel value if `frame_height` provided, else normalized.
                - xmax (float): Maximum X coordinate of the bounding box.
                                Pixel value if `frame_width` provided, else normalized.
                - ymax (float): Maximum Y coordinate of the bounding box.
                                Pixel value if `frame_height` provided, else normalized.
        
        Raises:
            AttributeError: If the `detection` object does not have the expected
                            `spatialCoordinates` or bounding box attributes.
        """
        # Ensure spatialCoordinates exist, otherwise raise AttributeError
        if not hasattr(detection, 'spatialCoordinates'):
            raise AttributeError("Detection object missing 'spatialCoordinates' attribute.")
        
        x = float(detection.spatialCoordinates.x)
        y = float(detection.spatialCoordinates.y)
        z = float(detection.spatialCoordinates.z)

        # Ensure bounding box attributes exist
        if not all(hasattr(detection, attr) for attr in ['xmin', 'xmax', 'ymin', 'ymax']):
            raise AttributeError("Detection object missing one or more bounding box attributes (xmin, xmax, ymin, ymax).")

        xmin_norm = float(detection.xmin)
        xmax_norm = float(detection.xmax)
        ymin_norm = float(detection.ymin)
        ymax_norm = float(detection.ymax)

        if frame_width is not None and frame_height is not None:
            xmin_pixel = int(xmin_norm * frame_width)
            xmax_pixel = int(xmax_norm * frame_width)
            ymin_pixel = int(ymin_norm * frame_height)
            ymax_pixel = int(ymax_norm * frame_height)
            return x, y, z, float(xmin_pixel), float(ymin_pixel), float(xmax_pixel), float(ymax_pixel)
        else:
            return x, y, z, xmin_norm, ymin_norm, xmax_norm, ymax_norm


    def get_roi_from_detection(self, detection: 'any', depth_frame_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """
        Extracts the Region of Interest (ROI) pixel coordinates from a detection's `boundingBoxMapping`.

        This method is typically used when detections provide ROI information relative
        to a specific frame (often the depth frame), and that ROI needs to be
        denormalized to pixel coordinates.

        Args:
            detection (any): A DepthAI detection object. Expected to have a
                `boundingBoxMapping` attribute, which in turn has an `roi` attribute.
                The `roi` attribute should be an object with a `denormalize` method
                (like `depthai.Rect`) that accepts frame width and height.
            depth_frame_shape (tuple[int, int]): Shape of the depth frame (height, width)
                used for denormalizing the ROI coordinates.

        Returns:
            tuple[int, int, int, int] | None: A tuple `(xmin, ymin, xmax, ymax)`
            representing the pixel coordinates of the ROI in the depth frame.
            Returns `None` if `boundingBoxMapping` or `roi` is not available in the
            detection object, or if denormalization fails.
            
        Raises:
            AttributeError: If `detection.boundingBoxMapping.roi` does not have a
                            `denormalize` method or `topLeft`/`bottomRight` methods
                            after denormalization.
        """
        if not hasattr(detection, 'boundingBoxMapping'):
            return None
        if not hasattr(detection.boundingBoxMapping, 'roi'):
            return None

        roi_data = detection.boundingBoxMapping
        # The `roi` object itself should have a denormalize method.
        # Example: depthai.Rect().denormalize(width, height)
        denormalized_roi = roi_data.roi.denormalize(depth_frame_shape[1], depth_frame_shape[0])
        
        # After denormalization, topLeft() and bottomRight() should yield points with x, y
        topLeft = denormalized_roi.topLeft()
        bottomRight = denormalized_roi.bottomRight()
        
        xmin = int(topLeft.x)
        ymin = int(topLeft.y)
        xmax = int(bottomRight.x)
        ymax = int(bottomRight.y)
        return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    print("SpatialProcessor conceptual examples:")
    processor = SpatialProcessor()

    # Example 1: Process a dummy depth frame
    print("\n--- Example 1: Process Dummy Depth Frame ---")
    dummy_depth_frame_data = (np.random.rand(400, 640) * 5000).astype(np.uint16) # Simulating mm values
    dummy_depth_frame_data[100:150, 200:250] = 1000 # Add a closer region
    dummy_depth_frame_data[0:10, 0:10] = 0 # Add some zero values

    colorized_depth, original_depth = processor.process_depth_frame(dummy_depth_frame_data.copy())
    if colorized_depth is not None:
        print(f"  Processed dummy depth frame. Colorized shape: {colorized_depth.shape}, Original shape: {original_depth.shape}")
        # cv2.imshow("Dummy Colorized Depth", colorized_depth)
        # cv2.waitKey(0) # Press any key to continue
        # cv2.destroyAllWindows()
    else:
        print("  Processing dummy depth frame failed.")


    # Example 2: Extract spatial coordinates from a mock detection
    print("\n--- Example 2: Extract Spatial Coordinates ---")
    class MockSpatialDetection:
        """Mocks a DepthAI spatial detection object."""
        def __init__(self, x, y, z, xmin_norm, ymin_norm, xmax_norm, ymax_norm):
            class SpatialCoords:
                pass
            self.spatialCoordinates = SpatialCoords()
            self.spatialCoordinates.x = x
            self.spatialCoordinates.y = y
            self.spatialCoordinates.z = z
            self.xmin = xmin_norm
            self.ymin = ymin_norm
            self.xmax = xmax_norm
            self.ymax = ymax_norm

    mock_det = MockSpatialDetection(x=100.5, y=50.2, z=1500.0, xmin_norm=0.1, ymin_norm=0.2, xmax_norm=0.3, ymax_norm=0.4)
    
    # With frame dimensions for denormalization
    coords_denorm = processor.get_spatial_coordinates(mock_det, frame_width=640, frame_height=480)
    print(f"  Extracted spatial coordinates (denormalized bbox): {coords_denorm}")
    # Expected: (100.5, 50.2, 1500.0, 64, 96, 192, 192)

    # Without frame dimensions (normalized bbox)
    coords_norm = processor.get_spatial_coordinates(mock_det)
    print(f"  Extracted spatial coordinates (normalized bbox): {coords_norm}")
    # Expected: (100.5, 50.2, 1500.0, 0.1, 0.2, 0.3, 0.4)

    # Example 3: Get ROI from a mock detection with boundingBoxMapping
    print("\n--- Example 3: Get ROI from Detection ---")
    class MockPoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockDenormalizedRect:
        def __init__(self, x, y, width, height, frame_w, frame_h):
            # Store normalized values and denormalize on demand
            self.norm_x = x
            self.norm_y = y
            self.norm_width = width
            self.norm_height = height
            self.frame_w = frame_w
            self.frame_h = frame_h

        def topLeft(self):
            return MockPoint(self.norm_x * self.frame_w, self.norm_y * self.frame_h)
        def bottomRight(self):
            return MockPoint((self.norm_x + self.norm_width) * self.frame_w, (self.norm_y + self.norm_height) * self.frame_h)

    class MockRect: # Simplified mock of depthai.Rect
        def __init__(self, x_norm, y_norm, width_norm, height_norm):
            self.x = x_norm
            self.y = y_norm
            self.width = width_norm
            self.height = height_norm
        def denormalize(self, frame_w, frame_h):
            # In a real scenario, depthai.Rect.denormalize returns a new depthai.Rect
            # Here, we simulate that the denormalized rect's topLeft/bottomRight are callable
            return MockDenormalizedRect(self.x, self.y, self.width, self.height, frame_w, frame_h)


    class MockDetectionWithRoi:
        """Mocks a detection object that has boundingBoxMapping."""
        def __init__(self, norm_x, norm_y, norm_width, norm_height):
            class BBoxMapping:
                pass
            self.boundingBoxMapping = BBoxMapping()
            self.boundingBoxMapping.roi = MockRect(norm_x, norm_y, norm_width, norm_height)

    mock_det_roi = MockDetectionWithRoi(norm_x=0.2, norm_y=0.3, norm_width=0.25, norm_height=0.35)
    mock_depth_shape_data = (400, 640) # height, width
    
    roi_coords = processor.get_roi_from_detection(mock_det_roi, mock_depth_shape_data)
    print(f"  Extracted ROI from mock detection for depth shape {mock_depth_shape_data}: {roi_coords}")
    # Expected: xmin = 0.2*640 = 128, ymin = 0.3*400 = 120
    #           xmax = (0.2+0.25)*640 = 0.45*640 = 288
    #           ymax = (0.3+0.35)*400 = 0.65*400 = 260
    # So, (128, 120, 288, 260)

    print("\nSpatialProcessor conceptual examples complete.")
    print("Full test requires integrating with DepthAI device and actual detections.")

```
