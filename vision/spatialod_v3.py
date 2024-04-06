#!/usr/bin/env python3
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import serial
import time
import threading

serial_port = '/dev/ttyUSB0'
baudrate = 9600
#ser = serial.Serial(serial_port, baudrate)

# Define serial port and baudrate
# serial_port = '/dev/ttyUSB0'  # Modify this to match your Arduino's serial port
# baudrate = 9600  # Modify this to match the baudrate of your Arduino's Serial.begin() function

# ser = serial.Serial(serial_port, baudrate)

# Get arguments
nnBlobPath = '/home/pi/Desktop/eyes_on_me/models/yolov7_v2/best_yolov7_v1_openvino_2022.1_6shave.blob'
syncNN = True

# Constants
DEGREES_PER_SECOND_TO_DEGREES_PER_SAMPLE = 0.01
# Variables
base_ts = None
reference_orientation = [0, 0, 0]  # Initial reference orientation

def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds()*1000

# Function to update reference orientation based on gyroscope data
def update_reference_orientation(gyro_values, delta_time):
    global reference_orientation
    reference_orientation[0] += gyro_values.x * delta_time
    reference_orientation[1] += gyro_values.y * delta_time
    reference_orientation[2] += gyro_values.z * delta_time

# Function to calculate rotation with respect to reference orientation
def calculate_rotation(gyro_values):
    rotation = [
        gyro_values.x - reference_orientation[0],
        gyro_values.y - reference_orientation[1],
        gyro_values.z - reference_orientation[2]
    ]
    return rotation

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)
imu = pipeline.create(dai.node.IMU)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xlinkGyro = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")
xlinkGyro.setStreamName("imu")

# Properties for RGB camera
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Properties for stereo camera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Properties for IMU
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xlinkGyro.input)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setSubpixel(True)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(6)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326])
spatialDetectionNetwork.setAnchorMasks({ "side80": [0,1,2], "side40": [3,4,5], "side20": [6,7,8] })
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)
spatialDetectionNetwork.out.link(xoutNN.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

time.sleep(1)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)
    imuQueue = device.getOutputQueue(name="imu", maxSize=1, blocking=False)
    baseTs = None

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True



    while True:

        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()
        imuData = imuQueue.get()

        imuPackets = imuData.packets

        for imuPacket in imuPackets:
            acceleroValues = imuPacket.acceleroMeter
            gyroValues = imuPacket.gyroscope

            acceleroTs = acceleroValues.getTimestampDevice()
            gyroTs = gyroValues.getTimestampDevice()

            if baseTs is None:
                baseTs = min(acceleroTs, gyroTs)
            acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
            gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

            # Convert gyroscope data to degrees per second
            gyro_values_degrees = dai.Gyroscope.floatToDeg(gyroValues)

            # Update reference orientation based on gyroscope data
            delta_time = (gyroTs - baseTs) / 1000  # Convert milliseconds to seconds
            update_reference_orientation(gyro_values_degrees, delta_time)

            # Calculate rotation with respect to reference orientation
            rotation = calculate_rotation(gyro_values_degrees)


            # imuF = "{:.06f}"
            # tsF  = "{:.03f}"
            #print(f"Accelerometer timestamp: {tsF.format(acceleroTs)} ms")
            #print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            #print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
            #print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")
            
            print("Rotation (degrees):", rotation)


        if printOutputLayersOnce:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters
        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:

            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            
            x = int(detection.spatialCoordinates.x)
            y = int(detection.spatialCoordinates.y)
            z = int(detection.spatialCoordinates.z)
            
            try:
                label_index = str(detection.label)
                #label = labelMap[label_index]
            except:
                label_index = str(detection.label)
                #label = LabelMap[label_index]
            cv2.putText(frame, str(label_index), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {x} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {y} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {z} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        # cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            serial.close()
            break
