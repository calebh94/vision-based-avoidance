'''


Script for Emergency Response algorithm



'''
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
        self.active = 'None'
        self.got_dnn = False
        self.got_astar = False
        self.restarting = False

        self.dnn = []
        self.astar = []

        self.active = 'dnn'

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
            if abs(self.pose_x - -5) < 0.01:
                self.restarting = False
                self.got_odom = False
                self.got_astar = False
                self.got_dnn = False
                self.astar = Point(0, 0, 1)
                self.dnn = Point(0, 0, 1)
                print('Returned to start position')
                print('Waiting to reset Astar and DNN')
                # time.sleep(5)

            else:
                wpt = MultiDOFJointTrajectory()
                header = Header()
                header.stamp = rospy.Time()
                header.frame_id = 'frame'
                wpt.joint_names.append('base_link')
                wpt.header = header
                quat = quaternion.from_euler_angles(0, 0, math.radians(-45))

                transforms = Transform(translation=Point(0,0,1), rotation=quat)

                velocities = Twist()
                accelerations = Twist()
                point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(2))
                wpt.points.append(point)
                self.waypoint_pub.publish(wpt)

        if abs(self.pose_x - 10) < 0.1:
            self.restarting = True
            print('Made it to goal location')

    def callback_dnn(self, msg_dnn):
        self.dnn = msg_dnn
        action = msg_dnn
        self.got_dnn = True

        if self.active == 'dnn':
            self.emergency_response(action)

    def callback_astar(self, msg_astar):
        self.astar = msg_astar
        action = msg_astar
        self.got_astar = True

        if self.active == 'astar':
            self.emergency_response(action)


    def emergency_response(self, action):
        pub_action = Point()
        # IN FUTURE CHANGE TO RELATIVE
        if abs(action.y - self.pose_y) < 0.05:
            pub_action.x = 5
            pub_action.y = 0
            pub_action.z = 1.5
        else:
            pub_action = action

        wpt = MultiDOFJointTrajectory()
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = 'frame'
        wpt.joint_names.append('base_link')
        wpt.header = header
        quat = quaternion.from_euler_angles(0, 0, math.radians(-45))
        #
        # if self.active == 'astar':
        #     transforms = Transform(translation=self.astar, rotation=quat)
        # elif self.active == 'dnn':
        #     transforms = Transform(translation=self.dnn, rotation=quat)
        # else:
        #     print('No Choice Made!')
        #     transforms = Transform()

        print('Selected {}'.format(self.active))

        transforms = Transform(translation=pub_action, rotation=quat)
        velocities = Twist()
        accelerations = Twist()
        point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(0))
        wpt.points.append(point)
        self.waypoint_pub.publish(wpt)



        #currently making choice here after Astar solution is received (ASSUMING SLOWER!)
        # CHECK WHICH IS SLOWER***
        # self.random_selection(

    # def random_selection(self):
    #     if self.got_dnn and self.got_astar and self.restarting is False:
    #
    #         choices = ['astar', 'dnn']
    #         prob = [1 - self.dagger_beta, self.dagger_beta]
    #
    #         self.active = np.random.choice(choices, p=prob)
    #
    #         wpt = MultiDOFJointTrajectory()
    #         header = Header()
    #         header.stamp = rospy.Time()
    #         header.frame_id = 'frame'
    #         wpt.joint_names.append('base_link')
    #         wpt.header = header
    #         quat = quaternion.from_euler_angles(0, 0, math.radians(-45))
    #
    #         if self.active == 'astar':
    #             transforms = Transform(translation=self.astar, rotation=quat)
    #         elif self.active == 'dnn':
    #             transforms = Transform(translation=self.dnn, rotation=quat)
    #         else:
    #             print('No Choice Made!')
    #             transforms = Transform()
    #
    #         print('Selected {}'.format(self.active))
    #
    #         velocities = Twist()
    #         accelerations = Twist()
    #         point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(2))
    #         wpt.points.append(point)
    #         self.waypoint_pub.publish(wpt)


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

