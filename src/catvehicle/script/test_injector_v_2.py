#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

def create_empty_point_cloud_message():
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "catvehicle/velodyne_link"
    msg.height = 1
    msg.width = 0
    msg.fields = []
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 0
    msg.data = []
    msg.is_dense = True
    return msg

def publish_test_data():
    pub = rospy.Publisher('/test_point_cloud', PointCloud2, queue_size=10)
    rospy.init_node('test_publisher', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz

    empty_data_duration = 5  # Duration for publishing empty data in seconds
    real_data_duration = 5   # Duration for publishing real data in seconds
    start_time = rospy.Time.now()

    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        elapsed_time = (current_time - start_time).to_sec()

        if elapsed_time < empty_data_duration:
            empty_msg = create_empty_point_cloud_message()
            pub.publish(empty_msg)
            rospy.loginfo("Published empty test point cloud message")
        else:
            start_time = rospy.Time.now()  # Reset the timer

        rate.sleep()

if __name__ == '__main__':
    publish_test_data()
