import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()

        # Create a publisher for the detected circles count
        self.circle_count_publisher = self.create_publisher(Int32, 'detected_circle_count', 10)

        # Initialize parameters with default values
        self.param1 = 100
        self.param2 = 30
        self.min_radius = 1
        self.max_radius = 30

    def detect_circles_from_realsense(self):
        # Create a pipeline
        pipeline = rs.pipeline()

        # Configure the pipeline for depth and color streaming
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the pipeline
        pipeline.start(config)

        try:
            while True:
                # Wait for the next set of frames from the camera
                frames = pipeline.wait_for_frames()

                # Get the depth and color frames
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert the color frame to a numpy array
                frame = np.asanyarray(color_frame.get_data())

                # Convert the frame to grayscale
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Apply median blur for noise reduction
                gray = cv.medianBlur(gray, 5)

                # Detect circles using HoughCircles with adjustable parameters
                circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, gray.shape[0] / 8,
                                          param1=self.param1, param2=self.param2,
                                          minRadius=self.min_radius, maxRadius=self.max_radius)

                if circles is not None:
                    circle_count = len(circles[0])
                    self.circle_count_publisher.publish(circle_count)

                # Sleep for a short while to control the loop rate
                rclpy.spin_once(self)

        finally:
            # Stop the pipeline
            pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    circle_detector = CircleDetector()
    circle_detector.detect_circles_from_realsense()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
