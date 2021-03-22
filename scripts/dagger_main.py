#!/home/charris/obstacle_avoidance/src/dagger_pytorch_ros/venv/bin/python

"""
Script for DAgger algorithm to select the waypoint using the expert or learned policy

by Caleb Harris (caleb.harris94@gatech.edu)
"""

import sys
import time

from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, Twist, Transform, Quaternion

import numpy as np
import math
import quaternion

import rospy
import roslib
roslib.load_manifest('rotors_gazebo')


class DAggerNode:

    def __init__(self, namespace):
        self.namespace = namespace
        self.start_time = rospy.Time.now()

        self.got_odom = False
        self.pose_x = []
        self.pose_y = []
        self.pose_z = []

        self.waypoint = MultiDOFJointTrajectoryPoint()
        self.got_dnn = False
        self.got_astar = False
        self.restarting = False
        self.training = True

        self.dnn = []
        self.astar = []

        self.active = 'astar'  # To Start
        self.choice_count = 10 #choose every 10 actions
        self.choice_current = 10
        self.dagger_beta = 0.0  # 1.0 ALL POLICY, 0.0 ALL EXPERT

        waypoint_string = '/' + str(self.namespace) + '/command/trajectory'
        selection_string = '/' + str(self.namespace + '/dagger/selection')
        odom_string = '/' + str(self.namespace) + '/ground_truth/odometry'
        dnn_string = '/' + str(self.namespace) + '/dnn/nextpoint'
        astar_string = '/' + str(self.namespace) + '/astar/nextpoint'

        self.waypoint_pub = rospy.Publisher(waypoint_string, MultiDOFJointTrajectory, queue_size=5)
        self.selection_pub = rospy.Publisher(selection_string, String, queue_size=5)
        self.odom_sub = rospy.Subscriber(odom_string, Odometry, self.callback_odom, queue_size=10)
        self.dnn_sub = rospy.Subscriber(dnn_string, Point, self.callback_dnn, queue_size=1)
        self.astar_sub = rospy.Subscriber(astar_string, Point, self.callback_astar, queue_size=1)

    def callback_odom(self, msg_odom):
        self.pose_x = msg_odom.pose.pose.position.x
        self.pose_y = msg_odom.pose.pose.position.y
        self.pose_z = msg_odom.pose.pose.position.z

        self.got_odom = True

        if self.restarting:
            self.astar = Point(0, 0, 1.5)
            self.dnn = Point(0, 0, 1.5)
            if abs(self.pose_x - 0) < 0.01:
                self.got_odom = False
                self.got_astar = False
                self.got_dnn = False
                print('Returned to start position')
                print('Waiting to reset Astar and DNN')
                rospy.sleep(3)
                self.restarting = False

            wpt = MultiDOFJointTrajectory()
            header = Header()
            header.stamp = rospy.Time()
            header.frame_id = 'frame'
            wpt.joint_names.append('base_link')
            wpt.header = header
            if self.namespace == "DJI":
                direction_theta = -45
            else:
                direction_theta = 0
            quat = quaternion.from_euler_angles(0, 0, math.radians(direction_theta))

            transforms = Transform(translation=Point(0,0,1.5), rotation=quat)

            velocities = Twist()
            accelerations = Twist()
            point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(2))
            wpt.points.append(point)
            self.waypoint_pub.publish(wpt)

        if abs(self.pose_x - 5) < 0.1 and self.training:
            self.restarting = True
            print('Made it to goal location')

    def callback_dnn(self, msg_dnn):
        if not self.restarting:
            self.dnn = msg_dnn
            self.got_dnn = True

        # self.random_selection()

    def callback_astar(self, msg_astar):
        if not self.restarting:
            self.astar = msg_astar
            self.got_astar = True

            self.random_selection()

    def random_selection(self):
        if self.got_dnn and self.got_astar and self.restarting is False:\
            # Choose every n times
            if self.choice_current == self.choice_count:
                choices = ['astar', 'dnn']
                prob = [1 - self.dagger_beta, self.dagger_beta]
                self.active = np.random.choice(choices, p=prob)
                self.choice_current = 1
            else:
                self.choice_current += 1

            wpt = MultiDOFJointTrajectory()
            header = Header()
            header.stamp = rospy.Time()
            header.frame_id = 'frame'
            wpt.joint_names.append('base_link')
            wpt.header = header
            quat = quaternion.from_euler_angles(0, 0, math.radians(-45))

            if self.active == 'astar':
                transforms = Transform(translation=self.astar, rotation=quat)
            elif self.active == 'dnn':
                transforms = Transform(translation=self.dnn, rotation=quat)
            else:
                print('No Choice Made!')
                transforms = Transform()

            print('Selected {}'.format(self.active))

            velocities = Twist()
            accelerations = Twist()
            point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(2))
            wpt.points.append(point)
            self.waypoint_pub.publish(wpt)


def dagger_main():
    if (len(sys.argv) > 1):
        namespace = sys.argv[1]
    else:
        namespace = 'DJI'
    rospy.loginfo("DAGGER Main process starting up! \n Please initiate expert and policy modules!\n ==== RUNNING ====")
    # print("DAGGER Main process starting up! \n Please initiate expert and policy modules!\n ==== RUNNING ====")

    while not rospy.is_shutdown():
        rospy.init_node('dagger')
        DAggerNode(namespace=namespace)
        rospy.spin()


if __name__ == '__main__':
    try:
        dagger_main()
    except rospy.ROSInterruptException:
        pass

