#!/usr/bin/env python3

import cv2
import argparse
import time
import depthai as dai
import numpy as np
import sys
from threading import Thread
import importlib.util

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils



# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras:', device.getConnectedCameraFeatures())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Device name
    print('Device name:', device.getDeviceName(), ' Product name:', device.getProductName())

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
      # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name='efficientdet_lite0.tflite', use_coral=True, num_threads=4)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        frame = inRgb.getCvFrame()
        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(frame)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

        if cv2.waitKey(1) == ord('q'):
            break
