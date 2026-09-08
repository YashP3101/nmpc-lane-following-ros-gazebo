#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image, PointField
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs import point_cloud2
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
from plc2pxl import pcl2pxl  # Custom function to convert 3D points to 2D pixels

class PointCloudToPointCloud2:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pointcloud_to_pointcloud2_converter')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create a PointCloud2 publisher
        self.publisher = rospy.Publisher('/output_point_cloud2', PointCloud2, queue_size=10)

        # Create Subscribers for the point cloud and image topics
        self.pc_sub = Subscriber('/catvehicle/scan', PointCloud2)
        self.image_sub = Subscriber('/catvehicle/camera_left/image_raw_left', Image)

        # Initialize an ApproximateTimeSynchronizer
        self.sync = ApproximateTimeSynchronizer([self.pc_sub, self.image_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)

    def synchronized_callback(self, pointcloud_msg, image_msg):
        # Convert the image message to a NumPy array
        latest_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Process point cloud data
        point_generator = point_cloud2.read_points(pointcloud_msg, skip_nans=True, field_names=('x', 'y', 'z'))

        # Get image dimensions for boundary checks
        img_height, img_width, _ = latest_image.shape

        # Prepare the list of points in the correct format for PointCloud2
        points = []

        # Process points
        for point in point_generator:
            x, y, z = point
            #z = -0.9269
            #z=-1.051878604
            
            # Convert 3D point to pixel coordinates (u, v)
            u, v = pcl2pxl(x, y, z)

            # Check if the pixel coordinates are within the image bounds
            if 0 <= u < img_width and 0 <= v < img_height:
                # Retrieve the color at pixel (u, v)
                color_bgr = latest_image[int(v), int(u)]

                # Convert BGR to RGBA
                color_rgba = ColorRGBA(
                    r=color_bgr[2] / 255.0,  # OpenCV uses BGR, ROS uses RGB
                    g=color_bgr[1] / 255.0,
                    b=color_bgr[0] / 255.0,
                    a=1.0  # Alpha value
                )

                # Convert RGBA to uint32 for PointCloud2
                rgba = self.rgba_to_uint32(color_rgba)

                # Add the point and its color data to the list
                points.append((x, y, z, rgba))

        # Create a PointCloud2 message
        header = pointcloud_msg.header
        header.frame_id = "catvehicle/velodyne_link"
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
        ]

        # Create a PointCloud2 message using the points
        cloud = point_cloud2.create_cloud(header, fields, points)
        
        # Publish the PointCloud2 message
        self.publisher.publish(cloud)

    def rgba_to_uint32(self, color):
        # Convert ColorRGBA to a single uint32 for PointCloud2
        rgba = (
            int(color.a * 255) << 24 |  # Alpha channel in most significant byte
            int(color.b * 255) << 16 |  # Blue channel
            int(color.g * 255) << 8 |   # Green channel
            int(color.r * 255)          # Red channel in least significant byte
        )
        return rgba

    def run(self):
        # Spin to keep the script running
        rospy.spin()

if __name__ == '__main__':
    # Create an instance of the converter and run it
    converter = PointCloudToPointCloud2()
    converter.run()
