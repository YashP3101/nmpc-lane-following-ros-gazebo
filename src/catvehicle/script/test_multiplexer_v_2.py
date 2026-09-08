#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import SetBool, SetBoolResponse

class DataMultiplexer:
    def __init__(self):
        rospy.init_node('data_multiplexer')
        self.real_data_sub = rospy.Subscriber('/output_point_cloud2', PointCloud2, self.handle_real_data)
        self.test_data_sub = rospy.Subscriber('/test_point_cloud', PointCloud2, self.handle_test_data)
        self.output_pub = rospy.Publisher('/output_point_cloud2_new', PointCloud2, queue_size=10)

        self.use_test_data = False
        self.latest_real_data = None
        rospy.Service('toggle_test_data', SetBool, self.handle_toggle_test_data)
        rospy.loginfo("Data multiplexer initialized")

    def handle_real_data(self, msg):
        self.latest_real_data = msg
        if not self.use_test_data:
            self.output_pub.publish(msg)

    def handle_test_data(self, msg):
        if self.use_test_data:
            self.output_pub.publish(msg)

    def handle_toggle_test_data(self, req):
        self.use_test_data = req.data
        rospy.loginfo("Toggled test data to {}".format(self.use_test_data))
        return SetBoolResponse(success=True, message="Toggled test data to {}".format(self.use_test_data))

if __name__ == '__main__':
    mux = DataMultiplexer()
    rospy.spin()
