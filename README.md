As the machine learning model here used "https://github.com/RangiLyu/nanodet".
Ignored other detected objects.
Drawn ROI around Detected Toilet bowl


About ROS node

sudo apt update
sudo apt install python3-pip
python3 -m pip install -U rclpy
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_package
cd ~/ros2_ws
colcon build
source ~/ros2_ws/install/setup.bash
ros2 run my_package circle_detector_node.py

