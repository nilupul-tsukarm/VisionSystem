import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import threading
import argparse
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String
import json

from nanodet import NanoDet

# Check OpenCV version
assert cv.__version__ >= "4.9.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale

def unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)

def vis(preds, res_img, letterbox_scale, fps=None):
    ret = res_img.copy()
    A_x_values = []
    A_y_values = []
    B_x_values = []
    B_y_values = []
    angle = []

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw bboxes and labels for toilet class only
    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        classid = pred[-1].astype(np.int32)

        if classes[classid] == 'toilet':
            # bbox
            xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
            cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

            # label
            label = "{:s}: {:.2f}".format(classes[classid], conf)
            cv.putText(ret, label, (xmin, ymax + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
            
            # Define ROI within the bounding box (adjust the percentages as needed)
            roi_xmin = int(xmin + 0.05 * (xmax - xmin))  # 10% offset from xmin left
            roi_ymin = int(ymin + 0.35 * (ymax - ymin))  # 10% offset from ymin  up
            roi_xmax = int(xmax - 0.05 * (xmax - xmin))  # 10% offset from xmax  right
            roi_ymax = int(ymax - 0.01 * (ymax - ymin))  # 10% offset from ymax  down

            # Draw a red rectangle around the ROI
            cv.rectangle(ret, (roi_xmin, roi_ymin), (roi_xmax, roi_ymax), (225, 0, 255), thickness=1)
   
            canny_low = 4
            canny_high = 150
           
            # Fit an ellipse only within the ROI
            roi_gray = cv.cvtColor(ret[roi_ymin:roi_ymax, roi_xmin:roi_xmax], cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(roi_gray, (3, 3), 0)
            edges = cv.Canny(blurred, canny_low, canny_high)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Concatenate arrays vertically
            numpy_vertical = np.vstack((blurred, edges))  # Use the same array twice for demonstration
            #cv.imshow('Numpy Vertical', numpy_vertical)


            min_points= 5

            # Filter out contours that represent complete ellipses
            for contour in contours:
                if len(contour) >= min_points:  # Minimum number of points required to fit an ellipse originl 5
                    # Fit an ellipse to the contour
                    ellipse = cv.fitEllipse(contour)

                    # Ensure that the ellipse's width and height are not zero to avoid ZeroDivisionError
                    if ellipse[1][0] != 0 and ellipse[1][1] != 0:
                        # Calculate contour area and ellipse area
                        contour_area = cv.contourArea(contour)
                        ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4

                        ratio = contour_area / ellipse_area
                        if 25000 > contour_area >= 12000:
                            if ratio > 0.2:  # Adjust this threshold as needed original 0.9
                                # Offset ellipse coordinates by the bounding box coordinates
                                ellipse = ((ellipse[0][0] + roi_xmin, ellipse[0][1] + roi_ymin ), ellipse[1], ellipse[2])
                                cv.ellipse(ret, ellipse, (0, 255, 0), 2)
                                
                                # Mark the center of the ellipse
                                cv.circle(ret, (int(ellipse[0][0]), int(ellipse[0][1])), 5, (55, 25, 255), -1)
                                cv.putText(ret, "{:.2f} Sq.cm".format(contour_area), (int(ellipse[0][0])+ 10, int(ellipse[0][1]) - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                # Draw the perpendicular axis to the ellipse's plane
                                angle_rad = np.deg2rad(ellipse[2])
                                axis_length = max(ellipse[1][0], ellipse[1][1])

                                cv.line(ret, (int(ellipse[0][0]), int(ellipse[0][1])-100), (int(ellipse[0][0]), int(ellipse[0][1])+100), (0, 0, 220), 2)  # fix horizontal red

                                # Calculate the distance between the endpoints of the line
                                line_length = np.sqrt((ellipse[0][0] - int(ellipse[0][0] - axis_length * np.sin(angle_rad))) ** 2 +
                                                    (ellipse[0][1] - int(ellipse[0][1] + axis_length * np.cos(angle_rad))) ** 2)
                                
                                # Draw the line 
                                cv.line(ret, (int(ellipse[0][0] - axis_length * np.sin(angle_rad)), int(ellipse[0][1] + axis_length * np.cos(angle_rad))),   # diameter blue
                                        (int(ellipse[0][0] + axis_length * np.sin(angle_rad)), int(ellipse[0][1] - axis_length * np.cos(angle_rad))),
                                        (255, 155, 0), 2)
                                
                                # Define the desired length of the line
                                line_length = 120  # Adjust as needed for your desired length

                                grid_color = (255, 255, 255)

                                # Calculate the change in x and y for the desired length
                                delta_x_length = line_length * np.sin(angle_rad)
                                delta_y_length = line_length * np.cos(angle_rad)

                                line_start = (int(ellipse[0][0] - delta_x_length)-50, int(ellipse[0][1] + delta_y_length))
                                line_end = (int(ellipse[0][0] + delta_x_length)-50, int(ellipse[0][1] - delta_y_length))
                                cv.line(ret, line_start, line_end, grid_color, 2)
                                
                                line_start = (int(ellipse[0][0] - delta_x_length)-100, int(ellipse[0][1] + delta_y_length))
                                line_end = (int(ellipse[0][0] + delta_x_length)-100, int(ellipse[0][1] - delta_y_length))
                                cv.line(ret, line_start, line_end, grid_color, 2)


                                line_start = (int(ellipse[0][0] - delta_x_length)+50, int(ellipse[0][1] + delta_y_length))
                                line_end = (int(ellipse[0][0] + delta_x_length)+50, int(ellipse[0][1] - delta_y_length))
                                cv.line(ret, line_start, line_end, grid_color, 2)
                                
                                line_start = (int(ellipse[0][0] - delta_x_length)+100, int(ellipse[0][1] + delta_y_length))
                                line_end = (int(ellipse[0][0] + delta_x_length)+100, int(ellipse[0][1] - delta_y_length))
                                cv.line(ret, line_start, line_end, grid_color, 2)


                                # Define the desired length of the perpendicular lines
                                perpendicular_length = 120  # Adjust as needed for your desired length

                                # Calculate the slope of the original line
                                original_slope = np.tan(angle_rad)

                                # Calculate the slope of the perpendicular line
                                perpendicular_slope = -1 / original_slope

                                # Calculate the change in x and y for the desired length for perpendicular lines
                                delta_x_perpendicular = perpendicular_length * np.cos(angle_rad)
                                delta_y_perpendicular = perpendicular_length * np.sin(angle_rad)

                                # Calculate the start and end points for the first perpendicular line
                                line_start_perpendicular_1 = (int(ellipse[0][0] - delta_x_perpendicular), int(ellipse[0][1] - delta_y_perpendicular))
                                line_end_perpendicular_1 = (int(ellipse[0][0] + delta_x_perpendicular), int(ellipse[0][1] + delta_y_perpendicular))
                                cv.line(ret, line_start_perpendicular_1, line_end_perpendicular_1, grid_color, 2)
                                
                                line_start_perpendicular_1 = (int(ellipse[0][0] - delta_x_perpendicular), int(ellipse[0][1] - delta_y_perpendicular)+100)
                                line_end_perpendicular_1 = (int(ellipse[0][0] + delta_x_perpendicular), int(ellipse[0][1] + delta_y_perpendicular)+100)
                                cv.line(ret, line_start_perpendicular_1, line_end_perpendicular_1, grid_color, 2)

                                line_start_perpendicular_1 = (int(ellipse[0][0] - delta_x_perpendicular), int(ellipse[0][1] - delta_y_perpendicular)+50)
                                line_end_perpendicular_1 = (int(ellipse[0][0] + delta_x_perpendicular), int(ellipse[0][1] + delta_y_perpendicular)+50)
                                cv.line(ret, line_start_perpendicular_1, line_end_perpendicular_1, grid_color, 2)

                                line_start_perpendicular_1 = (int(ellipse[0][0] - delta_x_perpendicular), int(ellipse[0][1] - delta_y_perpendicular)-100)
                                line_end_perpendicular_1 = (int(ellipse[0][0] + delta_x_perpendicular), int(ellipse[0][1] + delta_y_perpendicular)-100)
                                cv.line(ret, line_start_perpendicular_1, line_end_perpendicular_1, grid_color, 2)

                                line_start_perpendicular_1 = (int(ellipse[0][0] - delta_x_perpendicular), int(ellipse[0][1] - delta_y_perpendicular)-50)
                                line_end_perpendicular_1 = (int(ellipse[0][0] + delta_x_perpendicular), int(ellipse[0][1] + delta_y_perpendicular)-50)
                                cv.line(ret, line_start_perpendicular_1, line_end_perpendicular_1, grid_color, 2)

                                # Calculate the fraction of the distance along the line where you want to place the dot 
                                dot_fraction = 0.60

                                # Calculate the coordinates of the dot along the line
                                dot_x = int(ellipse[0][0] - (axis_length * np.sin(angle_rad)) * dot_fraction)
                                dot_y = int(ellipse[0][1] + (axis_length * np.cos(angle_rad)) * dot_fraction)

                                # Draw the dot on the line
                                cv.circle(ret, (dot_x, dot_y), radius=5, color=(0, 255, 255), thickness=-1)  # thickness=-1 fills the circle
                                
                                #print("A[x,y]", " ", abs(dot_x),",", abs(dot_y))
                                
                                if 480 < dot_y:
                                    print("Bad dot_y val ", dot_y)
                                    dot_y = 300

                                if 640 > dot_x and 480 > dot_y:
                                    #depth_value = depth_frame.get_distance(abs(dot_x), abs(dot_y))
                                    depth_value = 0
                                    depth_value_cm = abs(depth_value) * 100
                                

                                if depth_value_cm != 0:
                                    cv.putText(ret, "A {:.2f} cm".format(depth_value_cm), (dot_x + 10, dot_y - 10),
                                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
                                #print("Depth to A"," ", depth_value_cm)
                                
                                bdot_fraction = -0.60

                                # Calculate the coordinates of the dot along the line
                                bdot_x = int(ellipse[0][0] - (axis_length * np.sin(angle_rad)) * bdot_fraction)
                                bdot_y = int(ellipse[0][1] + (axis_length * np.cos(angle_rad)) * bdot_fraction)

                                # Draw the dot on the line
                                cv.circle(ret, (bdot_x, bdot_y), radius=5, color=(0, 255, 255), thickness=-1)  # thickness=-1 fills the circle
                                
                                #print("B dot_x", bdot_x," ", abs(bdot_x), "B dot_y", bdot_y," ", abs(bdot_y))
                                
                                if 480 < bdot_y:
                                    print("Bad bdot_y val ", bdot_y)
                                    bdot_y = 300    
                                
                                # Define the endpoints of the two lines
                                line1_start = (int(ellipse[0][0] - axis_length * np.sin(angle_rad)), int(ellipse[0][1] + axis_length * np.cos(angle_rad)))
                                line1_end = (int(ellipse[0][0] + axis_length * np.sin(angle_rad)), int(ellipse[0][1] - axis_length * np.cos(angle_rad)))
                                line2_start = (int(ellipse[0][0]), int(ellipse[0][1])-100)
                                line2_end = (int(ellipse[0][0]), int(ellipse[0][1])+100)

                                # Calculate the vectors representing the lines
                                vector1 = np.array([line1_end[0] - line1_start[0], line1_end[1] - line1_start[1]])
                                vector2 = np.array([line2_end[0] - line2_start[0], line2_end[1] - line2_start[1]])

                                # Calculate the dot product of the vectors
                                dot_product = np.dot(vector1, vector2)

                                # Calculate the magnitudes of the vectors
                                magnitude1 = np.linalg.norm(vector1)
                                magnitude2 = np.linalg.norm(vector2)

                                # Calculate the cosine of the angle between the lines using the dot product formula
                                cosine_angle = dot_product / (magnitude1 * magnitude2)

                                # Calculate the angle in radians
                                angle_radians = np.arccos(cosine_angle)

                                # Convert the angle from radians to degrees
                                angle_degrees = np.degrees(angle_radians)
                                
                                if angle_degrees > 90:
                                    angle_degrees = angle_degrees - 180
                                    #print("Angle: ", "{:.2f} deg".format(angle_degrees))
                                else:
                                    angle_degrees = angle_degrees
                                    #print("Angle: ", "{:.2f} deg".format(angle_degrees))
                                
                                cv.putText(ret, "{:.2f} deg".format(angle_degrees), (int(ellipse[0][0])+ 10, int(ellipse[0][1]) + 15),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                A_x_values.append(abs(dot_x))
                                A_y_values.append(abs(dot_y))
                                B_x_values.append(abs(bdot_x))
                                B_y_values.append(abs(bdot_y))
                                angle_degrees = round(angle_degrees, 2)
                                angle.append(angle_degrees)
            
            
    return ret, A_x_values, A_y_values, B_x_values, B_y_values, angle


class JSONPublisher(Node):
    def __init__(self):
        super().__init__('json_publisher')
        self.publisher_json = self.create_publisher(String, 'json_data', 10)
        self.publisher_image = self.create_publisher(Image, 'annotated_image', 10)
        timer_period = 0.2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.bridge = CvBridge()
        
        self.model = NanoDet(
            modelPath='object_detection_nanodet_2022nov.onnx',
            prob_threshold=0.35,
            iou_threshold=0.6,
            backend_id=cv.dnn.DNN_BACKEND_OPENCV,
            target_id=cv.dnn.DNN_TARGET_CPU
        )
 
    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        input_blob, letterbox_scale = letterbox(color_image)
        preds = self.model.infer(input_blob)
        annotated_image, A_x_values, A_y_values, B_x_values, B_y_values, angle = vis(preds, color_image, letterbox_scale)
        
        # Check if all required values are not None and not empty lists
        if A_x_values is not None and A_y_values is not None and B_x_values is not None and B_y_values is not None and angle is not None \
                and A_x_values and A_y_values and B_x_values and B_y_values and angle:
            # All parameters are available and not empty, proceed with publishing
            data = {
                "yaw": angle,
                "point_A": {
                    "x": A_x_values,
                    "y": A_y_values
                },
                "point_B": {
                    "x": B_x_values,
                    "y": B_y_values
                }
            }
            json_str = json.dumps(data)  # Convert dictionary to JSON formatted string
            msg_json = String()
            msg_json.data = json_str
            self.publisher_json.publish(msg_json)
            self.get_logger().info('Publishing JSON data: "%s"' % msg_json.data)
        else:
            # Skip publishing if any of the parameters are None or empty
            self.get_logger().info('Skipping JSON data publishing due to missing or empty parameters')

        center = (annotated_image.shape[1] // 2, (annotated_image.shape[0] // 2)+50)  # Center of the frame
        radius = 90  # Radius of the semicircle
        color = (0, 0, 255)  # Color of the semicircle (green)
        thickness = 2  # Thickness of the semicircle
        cv.ellipse(annotated_image, center, (radius, radius), 0, 0, 180, color, thickness)
        # Convert the annotated image to a ROS Image message
        msg_image = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        self.publisher_image.publish(msg_image)
        self.get_logger().info('Publishing annotated image')
              



def main(args=None):
    rclpy.init(args=args)
    json_publisher = JSONPublisher()
    rclpy.spin(json_publisher)
    json_publisher.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
