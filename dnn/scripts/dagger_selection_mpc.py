'''


Script for DAgger algorithm to select the expert of learned policy



'''
import sys
import time

from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust
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

        # self.waypoint = MultiDOFJointTrajectoryPoint()
        self.action = RollPitchYawrateThrust
        self.active = 'None'
        self.got_dnn = False
        self.got_mpc = False
        self.restarting = False

        self.dnn = RollPitchYawrateThrust()
        self.mpc = RollPitchYawrateThrust()

        self.dagger_beta = 1.0

        waypoint_string = '/' + str(self.namespace) + '/command/trajectory'
        selection_string = '/' + str(self.namespace + '/dagger/roll_pitch_yawrate_thrust')
        odom_string = '/' + str(self.namespace) + '/ground_truth/odometry'
        dnn_string = '/' + str(self.namespace) + '/dnn/roll_pitch_yawrate_thrust'
        mpc_string = '/' + str(self.namespace) + '/command/roll_pitch_yawrate_thrust'

        # self.waypoint_pub = rospy.Publisher(waypoint_string, Rol, queue_size=5)
        self.selection_pub = rospy.Publisher(selection_string, RollPitchYawrateThrust, queue_size=5)
        self.odom_sub = rospy.Subscriber(odom_string, Odometry, self.callback_odom, queue_size=10)
        self.dnn_sub = rospy.Subscriber(dnn_string, RollPitchYawrateThrust, self.callback_dnn, queue_size=1)
        self.mpc_sub = rospy.Subscriber(mpc_string, RollPitchYawrateThrust, self.callback_astar, queue_size=1)

    def callback_odom(self, msg_odom):
        self.pose_x = msg_odom.pose.pose.position.x
        self.pose_y = msg_odom.pose.pose.position.y
        self.pose_z = msg_odom.pose.pose.position.z

        self.got_odom = True

        # if self.restarting:
        #     if abs(self.pose_x - 0) < 0.05:
        #         self.restarting = False
        #         self.got_odom = False
        #         self.got_mpc = False
        #         self.got_dnn = False
        #         self.mpc = []
        #         self.dnn = []
        #         print('Returned to start position')
        #         print('Waiting to reset Astar and DNN')
        #         time.sleep(5)
        #
        #     else:
        #         wpt = MultiDOFJointTrajectory()
        #         header = Header()
        #         header.stamp = rospy.Time()
        #         header.frame_id = 'frame'
        #         wpt.joint_names.append('base_link')
        #         wpt.header = header
        #         quat = quaternion.from_euler_angles(0, 0, math.radians(-45))
        #
        #         transforms = Transform(translation=Point(0,0,1.5), rotation=quat)
        #
        #         velocities = Twist()
        #         accelerations = Twist()
        #         point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(2))
        #         wpt.points.append(point)
        #         self.waypoint_pub.publish(wpt)
        #
        # if abs(self.pose_x - 5) < 0.1:
        #     self.restarting = True
        #     print('Made it to goal location')

    def callback_dnn(self, msg_dnn):
        self.dnn = msg_dnn
        self.got_dnn = True

    def callback_astar(self, msg_mpc):
        self.mpc = msg_mpc
        self.got_mpc = True

        self.random_selection()

    def random_selection(self):
        if self.got_dnn and self.got_mpc and self.restarting is False:

            choices = ['mpc', 'dnn']
            prob = [1 - self.dagger_beta, self.dagger_beta]

            self.active = np.random.choice(choices, p=prob)

            print('Selected {}'.format(self.active))

            if self.active == 'mpc':
                self.action = self.mpc
            elif self.active == 'dnn':
                self.action = self.dnn
            else:
                print('NO CHOICE MADE!')

            self.selection_pub.publish(self.action)


def main():

    # rate = rospy.Rate(50) # 50 hz

    if (len(sys.argv) > 1):
        namespace = sys.argv[1]
    else:
        namespace = 'DJI'

    while not rospy.is_shutdown():
        rospy.init_node('DAGGER_Node')
        DAggerNode(namespace=namespace)
        rospy.spin()

    # while not rospy.is_shutdown():
    #     print('running')
    #     Dagger.random_selection()
    #     rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

