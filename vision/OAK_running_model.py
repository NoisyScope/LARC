import cv2
from depthai import Pipeline
import depthai as dai

# Define the path to your custom object detection model
model_path = "path/to/your/model.blob"

# Define the labels for your custom objects
labels = ["box_A", "box_QR_1", "box_blue","box_green","box_red","box_yellow"]  # Add your custom object labels

# Initialize the DepthAI pipeline
pipeline = Pipeline()

# Define the neural network that performs object detection
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(model_path)

# Set the input/output streams
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.preview.link(detection_nn.input)

# Set up depth calculation
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Set up depth calculation to the neural network
xout_rgb = pipeline.createXLinkOut()
xout_depth = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
cam_rgb.preview.link(xout_rgb.input)
stereo.depth.link(detection_nn.inputDepth)
detection_nn.passthroughDepth.link(xout_depth.input)

# Connect to the device
with dai.Device(pipeline) as device:
    # Start the pipeline
    device.startPipeline()

    # Define a function to calculate depth from disparity
    def calculate_depth(disparity):
        # Add your depth calculation logic here
        # This can be disparity-to-depth conversion or any other method you prefer
        return depth

    while True:
        # Get the next frame from the camera
        in_rgb = device.getOutputQueue("rgb").get()
        frame = in_rgb.getCvFrame()

        # Get the depth data
        depth_frame = device.getOutputQueue("depth").get()
        depth_data = depth_frame.getData()

        # Perform object detection
        detections = detection_nn.get().detections

        # Process the detections
        for detection in detections:
            # Get the bounding box coordinates
            bbox = (
                int(detection.xmin * frame.shape[1]),
                int(detection.ymin * frame.shape[0]),
                int(detection.xmax * frame.shape[1]),
                int(detection.ymax * frame.shape[0]),
            )

            # Draw the bounding box on the frame
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # Calculate the depth of the detected object
            depth = calculate_depth(depth_data[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            # Display the depth next to the object
            cv2.putText(frame, f"Depth: {depth} m", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()